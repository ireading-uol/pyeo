"""
rolling_composite_s2_change_detection
-------------------------------------
An app for providing continuous change detection based on classification of forest cover. Runs the following algorithm

 Step 1: Create an initial cloud-free median composite from Sentinel-2 as a baseline map

 Step 2: Download change detection images for the specific time window (L2A where available plus additional L1C).
         Preprocess all L1C images with Sen2Cor to make a cloud mask and atmospherically correct it to L2A.
         For each L2A image, get the directory paths of the separate band raster files.

 Step 3: Classify each L2A image and the baseline composite

 Step 4: Pair up successive classified images with the composite baseline map and identify all pixels with the change between classes of interest, e.g. from class 1 to 2,3 or 8

 Step 5: Update the baseline composite with the reflectance values of only the changed pixels. Update last_date of the baseline composite.

 Step 6: Create quicklooks.

 """
import shutil
import sys

import pyeo.classification
import pyeo.queries_and_downloads
import pyeo.raster_manipulation
import pyeo.filesystem_utilities
from pyeo.filesystem_utilities import get_filenames


import configparser
import copy
import argparse
import glob
import numpy as np
import os
from osgeo import gdal
import pandas as pd
import datetime as dt
from tempfile import TemporaryDirectory

gdal.UseExceptions()

def rolling_detection(config_path,
                      arg_start_date=None,
                      arg_end_date=None,
                      tile_id=None,
                      chunks=None,
                      build_composite=False,
                      do_download=False,
                      download_source="scihub",
                      build_prob_image=False,
                      do_classify=False,
                      do_update=False,
                      do_quicklooks=False,
                      do_delete=False
                      ):

    # If any processing step args are present, do not assume that we want to do all steps
    do_all = True
    if (build_composite or do_download or do_classify or do_update or do_delete or do_quicklooks) == True:
        do_all = False
    conf = configparser.ConfigParser(allow_no_value=True)
    conf.read(config_path)
    sen_user = conf['sent_2']['user']
    sen_pass = conf['sent_2']['pass']
    root_dir = conf['forest_sentinel']['root_dir']
    tile_root_dir = os.path.join(root_dir, tile_id)
    composite_start_date = conf['forest_sentinel']['composite_start']
    composite_end_date = conf['forest_sentinel']['composite_end']
    start_date = conf['forest_sentinel']['start_date']
    end_date = conf['forest_sentinel']['end_date']
    cloud_cover = conf['forest_sentinel']['cloud_cover']
    cloud_certainty_threshold = int(conf['forest_sentinel']['cloud_certainty_threshold'])
    model_path = conf['forest_sentinel']['model']
    sen2cor_path = conf['sen2cor']['path']
    epsg = int(conf['forest_sentinel']['epsg'])
   
    pyeo.filesystem_utilities.create_folder_structure_for_tiles(tile_root_dir)
    log = pyeo.filesystem_utilities.init_log(os.path.join(tile_root_dir, "log", tile_id+"_log.txt"))
    log.info("---------------------------------------------------------------")
    log.info("---                  PROCESSING START                       ---")
    log.info("---------------------------------------------------------------")
    log.info("Options:")
    if do_all:
        log.info("  --do_all")
    if build_composite:
        log.info("  --build_composite for baseline composite")
        log.info("  --download_source = {}".format(download_source))
    if do_download:
        log.info("  --do_download for change detection images")
        log.info("  --download_source = {}".format(download_source))
    if do_classify:
        log.info("  --do_classify to apply the random forest model and create classification layers")
    if build_prob_image:
        log.info("  --build_prob_image to save classification probability layers")
    if do_update:
        log.info("  --do_update to update the baseline composite with new observations")
    if do_quicklooks:
        log.info("  --do_quicklooks to create image quicklooks")
    if do_delete:
        log.info("  --do_delete to remove the downloaded L1C, L2A and cloud-masked composite layers after use")

    log.info("Creating the directory structure if not already present")

    try:
        l1_image_dir = os.path.join(tile_root_dir, r"images/L1C")
        l2_image_dir = os.path.join(tile_root_dir, r"images/L2A")
        l2_masked_image_dir = os.path.join(tile_root_dir, r"images/cloud_masked")
        categorised_image_dir = os.path.join(tile_root_dir, r"output/classified")
        probability_image_dir = os.path.join(tile_root_dir, r"output/probabilities")
        composite_dir = os.path.join(tile_root_dir, r"composite")
        composite_l1_image_dir = os.path.join(tile_root_dir, r"composite/L1C")
        composite_l2_image_dir = os.path.join(tile_root_dir, r"composite/L2A")
        composite_l2_masked_image_dir = os.path.join(tile_root_dir, r"composite/cloud_masked")
        quicklook_dir = os.path.join(tile_root_dir, r"output/quicklooks")

        if arg_start_date == "LATEST":
            # Returns the yyyymmdd string of the latest classified image
            start_date = pyeo.filesystem_utilities.get_image_acquisition_time(
                pyeo.filesystem_utilities.sort_by_timestamp(
                    [image_name for image_name in os.listdir(categorised_image_dir) if image_name.endswith(".tif")],
                    recent_first=True
                )[0]).strftime("%Y%m%d")
        elif arg_start_date:
            start_date = arg_start_date

        if arg_end_date == "TODAY":
            end_date = dt.date.today().strftime("%Y%m%d")
        elif arg_end_date:
            end_date = arg_end_date

        # ------------------------------------------------------------------------
        # Step 1: Create an initial cloud-free median composite from Sentinel-2 as a baseline map
        # ------------------------------------------------------------------------

        #TODO: Make the download optional at the compositing stage, i.e. if do_download is not selected, skip it 
        #      and only call the median compositing function. Should be a piece of cake.
        # if build_composite or do_all:
        #     if do_download or do_all:
        #         [...download the data for the composite...]
        #     [...calculate the median composite from the available data...]
        if build_composite or do_all:
            log.info("---------------------------------------------------------------")
            log.info("Creating an initial cloud-free median composite from Sentinel-2 as a baseline map")
            log.info("---------------------------------------------------------------")
            log.info("Searching for images for initial composite.")
 
            ''' 
            # could use this as a shortcut
            
            test1 = api.query(tileid = tile_id, platformname = 'Sentinel-2', processinglevel = 'Level-1C')
            test2 = api.query(tileid = tile_id, platformname = 'Sentinel-2', processinglevel = 'Level-2A')
            
            '''

            composite_products_all = pyeo.queries_and_downloads.check_for_s2_data_by_date(root_dir,
                                                                                          composite_start_date,
                                                                                          composite_end_date,
                                                                                          conf, 
                                                                                          cloud_cover=cloud_cover,
                                                                                          tile_id=tile_id,
                                                                                          producttype=None #"S2MSI2A" or "S2MSI1C"
                                                                                          )

            #TODO: retrieve metadata on nodata percentage and prioritise download of images with low values
            # This method currently only works for L2A products and needs expanding to L1C
            '''
            composite_products_all = pyeo.queries_and_downloads.get_nodata_percentage(sen_user, sen_pass, composite_products_all)
            log.info("NO_DATA_PERCENTAGE:")
            for uuid, metadata in composite_products_all.items():
                log.info("{}: {}".format(metadata['title'], metadata['No_data_percentage']))
            '''

            log.info("--> Found {} L1C and L2A products for the composite:".format(len(composite_products_all)))
            df_all = pd.DataFrame.from_dict(composite_products_all, orient='index')

            # check granule sizes on the server
            df_all['size'] = df_all['size'].str.split(' ').apply(lambda x: float(x[0]) * {'GB': 1e3, 'MB': 1, 'KB': 1e-3}[x[1]])
            df = df_all.query('size >= '+str(faulty_granule_threshold))
            log.info("Removed {} faulty scenes <{}MB in size from the list:".format(len(df_all)-len(df), faulty_granule_threshold))
            df_faulty = df_all.query('size < '+str(faulty_granule_threshold))
            for r in range(len(df_faulty)):
                log.info("   {} MB: {}".format(df_faulty.iloc[r,:]['size'], df_faulty.iloc[r,:]['title']))

            l1c_products = df[df.processinglevel == 'Level-1C']
            l2a_products = df[df.processinglevel == 'Level-2A']
            log.info("    {} L1C products".format(l1c_products.shape[0]))
            log.info("    {} L2A products".format(l2a_products.shape[0]))

            # during compositing stage, limit the number of images to download
            # to avoid only downloading partially covered granules with low cloud cover (which is calculated over the whole granule, 
            # incl. missing values), we need to stratify our search for low cloud cover by relative orbit number

            rel_orbits = np.unique(l1c_products['relativeorbitnumber'])
            if len(rel_orbits) > 0:
                if l1c_products.shape[0] > max_image_number/len(rel_orbits):
                    log.info("Capping the number of L1C products to {}".format(max_image_number))
                    log.info("Relative orbits found covering tile: {}".format(rel_orbits))
                    uuids = []
                    for orb in rel_orbits:
                        uuids = uuids + list(l1c_products.loc[l1c_products['relativeorbitnumber'] == orb].sort_values(by=['cloudcoverpercentage'], ascending=True)['uuid'][:int(max_image_number/len(rel_orbits))])
                    l1c_products = l1c_products[l1c_products['uuid'].isin(uuids)]
                    log.info("    {} L1C products remain:".format(l1c_products.shape[0]))
                    for product in l1c_products['title']:
                        log.info("       {}".format(product))

            rel_orbits = np.unique(l2a_products['relativeorbitnumber'])
            if len(rel_orbits) > 0:
                if l2a_products.shape[0] > max_image_number/len(rel_orbits):
                    log.info("Capping the number of L2A products to {}".format(max_image_number))
                    log.info("Relative orbits found covering tile: {}".format(rel_orbits))
                    uuids = []
                    for orb in rel_orbits:
                        uuids = uuids + list(l2a_products.loc[l2a_products['relativeorbitnumber'] == orb].sort_values(by=['cloudcoverpercentage'], ascending=True)['uuid'][:int(max_image_number/len(rel_orbits))])
                    l2a_products = l2a_products[l2a_products['uuid'].isin(uuids)]
                    log.info("    {} L2A products remain:".format(l2a_products.shape[0]))
                    for product in l2a_products['title']:
                        log.info("       {}".format(product))

            if l1c_products.shape[0]>0 and l2a_products.shape[0]>0:
                log.info("Filtering out L1C products that have the same 'beginposition' time stamp as an existing L2A product.")
                l1c_products, l2a_products = pyeo.queries_and_downloads.filter_unique_l1c_and_l2a_data(df)
                log.info("--> {} L1C and L2A products with unique 'beginposition' time stamp for the composite:".format(l1c_products.shape[0]+l2a_products.shape[0]))
                log.info("    {} L1C products".format(l1c_products.shape[0]))
                log.info("    {} L2A products".format(l2a_products.shape[0]))
            df = None

            #TODO: Before the next step, search the composite/L2A and L1C directories whether the scenes have already been downloaded and/or processed and check their dir sizes
            # Remove those already obtained from the list

            if l1c_products.shape[0] > 0:
                log.info("Checking for availability of L2A products to minimise download and atmospheric correction of L1C products.")
                n = len(l1c_products)
                drop=[]
                add=[]
                for r in range(n):
                    id = l1c_products.iloc[r,:]['title']
                    search_term = "*"+id.split("_")[2]+"_"+id.split("_")[3]+"_"+id.split("_")[4]+"_"+id.split("_")[5]+"*"
                    log.info("Search term: {}.".format(search_term))
                    matching_l2a_products = pyeo.queries_and_downloads._file_api_query(user=sen_user, 
                                                                                       passwd=sen_pass, 
                                                                                       start_date=composite_start_date,
                                                                                       end_date=composite_end_date,
                                                                                       filename=search_term,
                                                                                       cloud=cloud_cover,
                                                                                       producttype="S2MSI2A"
                                                                                       )

                    matching_l2a_products_df = pd.DataFrame.from_dict(matching_l2a_products, orient='index')
                    if len(matching_l2a_products_df) == 1 and matching_l2a_products_df.iloc[0,:]['size'].str.split(' ').apply(lambda x: float(x[0]) * {'GB': 1e3, 'MB': 1, 'KB': 1e-3}[x[1]]) > faulty_granule_threshold:
                        log.info("Replacing L1C {} with L2A product:".format(id))
                        log.info("              {}".format(matching_l2a_products_df.iloc[0,:]['title']))
                        drop.append(l1c_products.index[r])
                        add.append(matching_l2a_products_df.iloc[0,:])
                    if len(matching_l2a_products_df) == 0:
                        log.info("Found no match for L1C: {}.".format(id))
                    if len(matching_l2a_products_df) > 1:
                        # check granule sizes on the server
                        matching_l2a_products_df['size'] = matching_l2a_products_df['size'].str.split(' ').apply(lambda x: float(x[0]) * {'GB': 1e3, 'MB': 1, 'KB': 1e-3}[x[1]])
                        matching_l2a_products_df = matching_l2a_products_df.query('size >= '+str(faulty_granule_threshold))
                        if matching_l2a_products_df.iloc[0,:]['size'].str.split(' ').apply(lambda x: float(x[0]) * {'GB': 1e3, 'MB': 1, 'KB': 1e-3}[x[1]]) > faulty_granule_threshold:
                            log.info("Replacing L1C {} with L2A product:".format(id))
                            log.info("              {}".format(matching_l2a_products_df.iloc[0,:]['title']))
                            drop.append(l1c_products.index[r])
                            add.append(matching_l2a_products_df.iloc[0,:])
                if len(drop) > 0:
                    l1c_products = l1c_products.drop(index=drop)
                if len(add) > 0:
                    l2a_products = l2a_products.append(add)
                l2a_products = l2a_products.drop_duplicates(subset='title')
                log.info("    {} L1C products remaining for download".format(l1c_products.shape[0]))
                log.info("    {} L2A products remaining for download".format(l2a_products.shape[0]))

                log.info("Downloading Sentinel-2 L1C products.")
                #TODO: Need to collect the response from download_from_scihub function and check whether the download succeeded
                pyeo.queries_and_downloads.download_s2_data_from_df(l1c_products,
                                                            composite_l1_image_dir, 
                                                            composite_l2_image_dir, 
                                                            download_source,
                                                            user=sen_user, 
                                                            passwd=sen_pass, 
                                                            try_scihub_on_fail=True)
                log.info("Atmospheric correction with sen2cor.")
                pyeo.raster_manipulation.atmospheric_correction(composite_l1_image_dir, 
                                                                composite_l2_image_dir,
                                                                sen2cor_path,
                                                                delete_unprocessed_image=False)
            if l2a_products.shape[0] > 0:
                log.info("Downloading Sentinel-2 L2A products.")
                pyeo.queries_and_downloads.download_s2_data(l2a_products.to_dict('index'),
                                                            composite_l1_image_dir, 
                                                            composite_l2_image_dir, 
                                                            download_source,
                                                            user=sen_user, 
                                                            passwd=sen_pass, 
                                                            try_scihub_on_fail=True)

            # check for incomplete L2A downloads and remove them
            incomplete_downloads, sizes = pyeo.raster_manipulation.find_small_safe_dirs(composite_l2_image_dir, threshold=faulty_granule_threshold*1024*1024)
            if len(incomplete_downloads) > 0:
                for index, safe_dir in enumerate(incomplete_downloads):
                    if sizes[index]/1024/1024 < faulty_granule_threshold and os.path.exists(safe_dir):
                        log.warning("Found likely incomplete download of size {} MB: {}".format(str(round(sizes[index]/1024/1024)), safe_dir))
                        #shutil.rmtree(safe_dir)

            log.info("---------------------------------------------------------------")
            log.info("Image download and atmospheric correction for composite is complete.")
            log.info("---------------------------------------------------------------")

            l2a_paths = [ f.path for f in os.scandir(composite_l2_image_dir) if f.is_dir() ]
            #raster_paths = pyeo.filesystem_utilities.get_raster_paths(l2a_paths, filepatterns=bands, dirpattern=resolution) # don't really need to know these
            #scl_raster_paths = pyeo.filesystem_utilities.get_raster_paths(l2a_paths, filepatterns=["SCL"], dirpattern="20m") # don't really need to know these

            log.info("Applying simple cloud, cloud shadow and haze mask based on SCL files and stacking the masked band raster files.")
            pyeo.raster_manipulation.apply_scl_cloud_mask(composite_l2_image_dir, 
                                                          composite_l2_masked_image_dir, 
                                                          scl_classes=[0,1,2,3,8,9,10,11],
                                                          buffer_size=buffer_size_composite, 
                                                          bands=bands, 
                                                          out_resolution=10,
                                                          haze=None,
                                                          epsg=epsg,
                                                          skip_existing=skip_existing)

            log.info("Building initial cloud-free median composite from directory {}".format(composite_l2_masked_image_dir))
            pyeo.raster_manipulation.clever_composite_directory(composite_l2_masked_image_dir, 
                                                                composite_dir, 
                                                                chunks=chunks,
                                                                generate_date_images=True,
                                                                missing_data_value=0)

            log.info("---------------------------------------------------------------")
            log.info("Baseline image composite is complete.")
            log.info("---------------------------------------------------------------")

        # ------------------------------------------------------------------------
        # Step 2: Download change detection images for the specific time window (L2A where available plus additional L1C)
        # ------------------------------------------------------------------------
        if do_all or do_download:
            log.info("---------------------------------------------------------------")
            log.info("Downloading change detection images between {} and {} with cloud cover <= {}".format(
                     start_date, end_date, cloud_cover))
            log.info("---------------------------------------------------------------")

            products_all = pyeo.queries_and_downloads.check_for_s2_data_by_date(root_dir,
                                                                               start_date,
                                                                               end_date,
                                                                               conf, 
                                                                               cloud_cover=cloud_cover,
                                                                               tile_id=tile_id,
                                                                               producttype=None #"S2MSI2A" or "S2MSI1C"
                                                                               )
            log.info("--> Found {} L1C and L2A products for change detection:".format(len(products_all)))
            df_all = pd.DataFrame.from_dict(products_all, orient='index')

            # check granule sizes on the server
            df_all['size'] = df_all['size'].str.split(' ').apply(lambda x: float(x[0]) * {'GB': 1e3, 'MB': 1, 'KB': 1e-3}[x[1]])
            df = df_all.query('size >= '+str(faulty_granule_threshold))
            log.info("Removed {} faulty scenes <{}MB in size from the list:".format(len(df_all)-len(df), faulty_granule_threshold))
            df_faulty = df_all.query('size < '+str(faulty_granule_threshold))
            for r in range(len(df_faulty)):
                log.info("   {} MB: {}".format(df_faulty.iloc[r,:]['size'], df_faulty.iloc[r,:]['title']))

            l1c_products = df[df.processinglevel == 'Level-1C']
            l2a_products = df[df.processinglevel == 'Level-2A']
            log.info("    {} L1C products".format(l1c_products.shape[0]))
            log.info("    {} L2A products".format(l2a_products.shape[0]))

            if l1c_products.shape[0]>0 and l2a_products.shape[0]>0:
                log.info("Filtering out L1C products that have the same 'beginposition' time stamp as an existing L2A product.")
                l1c_products, l2a_products = pyeo.queries_and_downloads.filter_unique_l1c_and_l2a_data(df)
                log.info("--> {} L1C and L2A products with unique 'beginposition' time stamp for the composite:".format(l1c_products.shape[0]+l2a_products.shape[0]))
                log.info("    {} L1C products".format(l1c_products.shape[0]))
                log.info("    {} L2A products".format(l2a_products.shape[0]))
            df = None

            #TODO: Before the next step, seacrh the composite/L2A and L1C directories whether the scenes have already been downloaded and/or processed and check their dir sizes
            # Remove those already obtained from the list

            if l1c_products.shape[0] > 0:
                log.info("Checking for availability of L2A products to minimise download and atmospheric correction of L1C products.")
                n = len(l1c_products)
                drop=[]
                add=[]
                for r in range(n):
                    id = l1c_products.iloc[r,:]['title']
                    search_term = "*"+id.split("_")[2]+"_"+id.split("_")[3]+"_"+id.split("_")[4]+"_"+id.split("_")[5]+"*"
                    log.info("Search term: {}.".format(search_term))
                    matching_l2a_products = pyeo.queries_and_downloads._file_api_query(user=sen_user, 
                                                                                       passwd=sen_pass, 
                                                                                       start_date=start_date,
                                                                                       end_date=end_date,
                                                                                       filename=search_term,
                                                                                       cloud=cloud_cover,
                                                                                       producttype="S2MSI2A"
                                                                                       )

                    matching_l2a_products_df = pd.DataFrame.from_dict(matching_l2a_products, orient='index')
                    if len(matching_l2a_products_df) == 1:
                        log.info(matching_l2a_products_df.iloc[0,:]['size'])
                        matching_l2a_products_df['size'] = matching_l2a_products_df['size'].str.split(' ').apply(lambda x: float(x[0]) * {'GB': 1e3, 'MB': 1, 'KB': 1e-3}[x[1]])
                        if matching_l2a_products_df.iloc[0,:]['size'] > faulty_granule_threshold:
                            log.info("Replacing L1C {} with L2A product:".format(id))
                            log.info("              {}".format(matching_l2a_products_df.iloc[0,:]['title']))
                            drop.append(l1c_products.index[r])
                            add.append(matching_l2a_products_df.iloc[0,:])
                    if len(matching_l2a_products_df) == 0:
                        log.info("Found no match for L1C: {}.".format(id))
                    if len(matching_l2a_products_df) > 1:
                        # check granule sizes on the server
                        matching_l2a_products_df['size'] = matching_l2a_products_df['size'].str.split(' ').apply(lambda x: float(x[0]) * {'GB': 1e3, 'MB': 1, 'KB': 1e-3}[x[1]])
                        if matching_l2a_products_df.iloc[0,:]['size'] > faulty_granule_threshold:
                            log.info("Replacing L1C {} with L2A product:".format(id))
                            log.info("              {}".format(matching_l2a_products_df.iloc[0,:]['title']))
                            drop.append(l1c_products.index[r])
                            add.append(matching_l2a_products_df.iloc[0,:])

                if len(drop) > 0:
                    l1c_products = l1c_products.drop(index=drop)
                if len(add) > 0:
                    l2a_products = l2a_products.append(add)
                log.info("    {} L1C products remaining for download".format(l1c_products.shape[0]))
                log.info("    {} L2A products remaining for download".format(l2a_products.shape[0]))
                l2a_products = l2a_products.drop_duplicates(subset='title')
                log.info("Downloading Sentinel-2 L1C products.")
                pyeo.queries_and_downloads.download_s2_data_from_df(l1c_products,
                                                            l1_image_dir, 
                                                            l2_image_dir, 
                                                            download_source,
                                                            user=sen_user, 
                                                            passwd=sen_pass, 
                                                            try_scihub_on_fail=True)
                log.info("Atmospheric correction with sen2cor.")
                pyeo.raster_manipulation.atmospheric_correction(l1_image_dir, 
                                                                l2_image_dir,
                                                                sen2cor_path,
                                                                delete_unprocessed_image=False)
            if l2a_products.shape[0] > 0:
                log.info("Downloading Sentinel-2 L2A products.")
                pyeo.queries_and_downloads.download_s2_data(l2a_products.to_dict('index'),
                                                            l1_image_dir, 
                                                            l2_image_dir, 
                                                            download_source,
                                                            user=sen_user, 
                                                            passwd=sen_pass, 
                                                            try_scihub_on_fail=True)

            # check for incomplete L2A downloads and remove them
            incomplete_downloads, sizes = pyeo.raster_manipulation.find_small_safe_dirs(l2_image_dir, threshold=faulty_granule_threshold*1024*1024)
            if len(incomplete_downloads) > 0:
                for index, safe_dir in enumerate(incomplete_downloads):
                    if sizes[index]/1024/1024 < faulty_granule_threshold and os.path.exists(safe_dir):
                        log.warning("Found likely incomplete download of size {} MB: {}".format(str(round(sizes[index]/1024/1024)), safe_dir))
                        #shutil.rmtree(safe_dir)

            log.info("---------------------------------------------------------------")
            log.info("Image download and atmospheric correction for change detection images is complete.")
            log.info("---------------------------------------------------------------")
            log.info("Applying simple cloud, cloud shadow and haze mask based on SCL files and stacking the masked band raster files.")
            l2a_paths = [ f.path for f in os.scandir(l2_image_dir) if f.is_dir() ]
            log.info("  l2_image_dir: {}".format(l2_image_dir))
            log.info("  l2_masked_image_dir: {}".format(l2_masked_image_dir))
            log.info("  bands: {}".format(bands))
            pyeo.raster_manipulation.apply_scl_cloud_mask(l2_image_dir, 
                                                          l2_masked_image_dir, 
                                                          scl_classes=[0,1,2,3,8,9,10,11],
                                                          buffer_size=buffer_size, 
                                                          bands=bands, 
                                                          out_resolution=10,
                                                          haze=None,
                                                          epsg=epsg,
                                                          skip_existing=skip_existing)


        # ------------------------------------------------------------------------
        # Step 3: Classify each L2A image and the baseline composite
        # ------------------------------------------------------------------------
        if do_all or do_classify:
            log.info("---------------------------------------------------------------")
            log.info("Classify a land cover map for each L2A image and composite image using a saved model")
            log.info("---------------------------------------------------------------")
            log.info("Model used: {}".format(model_path))
            if skip_existing:
                log.info("Skipping existing classification images if found.") 
            pyeo.classification.classify_directory(composite_dir,
                                                   model_path,
                                                   categorised_image_dir,
                                                   prob_out_dir = None, 
                                                   apply_mask=False, 
                                                   out_type="GTiff", 
                                                   chunks=chunks,
                                                   skip_existing=skip_existing)
            pyeo.classification.classify_directory(l2_masked_image_dir,
                                                   model_path,
                                                   categorised_image_dir,
                                                   prob_out_dir = None, 
                                                   apply_mask=False, 
                                                   out_type="GTiff", 
                                                   chunks=chunks,
                                                   skip_existing=skip_existing)
            log.info("End of classification.")

        # ------------------------------------------------------------------------
        # Step 4: Pair up the class images with the composite baseline map 
        # and identify all pixels with the change between groups of classes of interest.
        # Currently this is being done together with the classification step.
        # ------------------------------------------------------------------------
        if do_all or do_classify:
            log.info("---------------------------------------------------------------")
            log.info("Creating change layers from stacked class images.")
            log.info("---------------------------------------------------------------")
            log.info("Change of interest is from any of the classes {} to any of the classes {}.".format(from_classes, to_classes))

            # get all image paths in the classification maps directory except the class composites
            class_image_paths = [ f.path for f in os.scandir(categorised_image_dir) if f.is_file() and f.name.endswith(".tif") \
                                  and not "composite_" in f.name ]
            if len(class_image_paths) == 0:
                raise FileNotFoundError("No class images found in {}.".format(categorised_image_dir))

            # sort class images by image acquisition date
            class_image_paths = list(filter(pyeo.filesystem_utilities.get_image_acquisition_time, class_image_paths))
            class_image_paths.sort(key=lambda x: pyeo.filesystem_utilities.get_image_acquisition_time(x))
            for index, image in enumerate(class_image_paths):
                log.info("{}: {}".format(index, image))

            # find the latest available composite
            try:
                latest_composite_name = \
                    pyeo.filesystem_utilities.sort_by_timestamp(
                        [image_name for image_name in os.listdir(composite_dir) if image_name.endswith(".tif")],
                        recent_first=True
                    )[0]
                latest_composite_path = os.path.join(composite_dir, latest_composite_name)
                log.info("Most recent composite at {}".format(latest_composite_path))
            except IndexError:
                log.critical("Latest composite not found. The first time you run this script, you need to include the "
                             "--build-composite flag to create a base composite to work off. If you have already done this,"
                             "check that the earliest dated image in your images/merged folder is later than the earliest"
                             " dated image in your composite/ folder.")
                sys.exit(1)
            latest_class_composite_path = os.path.join(categorised_image_dir, os.path.basename(latest_composite_path)[:-4]+"_class.tif")
            log.info("Most recent class composite at {}".format(latest_class_composite_path))
            if not os.path.exists(latest_class_composite_path):
                log.critical("Latest class composite not found. The first time you run this script, you need to include the "
                             "--build-composite flag to create a base composite to work off. If you have already done this,"
                             "check that the earliest dated image in your images/merged folder is later than the earliest"
                             " dated image in your composite/ folder. Then, you need to run the --classify option.")
                sys.exit(1)
            log.info("Latest class composite: {}".format(latest_class_composite_path))

            # find change patterns in the stack of classification images
            for index, image in enumerate(class_image_paths):
                before_timestamp = pyeo.filesystem_utilities.get_change_detection_dates(os.path.basename(latest_class_composite_path))[0]
                after_timestamp  = pyeo.filesystem_utilities.get_image_acquisition_time(os.path.basename(image))
                log.info("  early time stamp: {}".format(before_timestamp))
                log.info("  late  time stamp: {}".format(after_timestamp))
                change_raster = os.path.join(probability_image_dir,
                                             "change_{}_{}_{}.tif".format(
                                             before_timestamp.strftime("%Y%m%dT%H%M%S"),
                                             tile_id,
                                             after_timestamp.strftime("%Y%m%dT%H%M%S"))
                                             )
                log.info("  Change raster file to be created: {}".format(change_raster))
                # This function looks for changes from class 'change_from' in the composite to any of the 'change_to_classes'
                # in the change images. Pixel values are the acquisition date of the detected change of interest or zero.
                pyeo.raster_manipulation.change_from_class_maps(latest_class_composite_path,
                                                                image,
                                                                change_raster, 
                                                                change_from = from_classes,
                                                                change_to = to_classes,
                                                                skip_existing = skip_existing)

            # combine all change layers into one output raster with two layers:
            #   (1) pixels show the earliest change detection date (expressed as the number of days since 1/1/2000)
            #   (2) pixels show the number of change detection dates (summed up over all change images in the folder)
            date_image_paths = [ f.path for f in os.scandir(probability_image_dir) if f.is_file() and f.name.endswith(".tif") \
                                 and "change_" in f.name ]
            if len(date_image_paths) == 0:
                raise FileNotFoundError("No class images found in {}.".format(categorised_image_dir))

            before_timestamp = pyeo.filesystem_utilities.get_change_detection_dates(os.path.basename(latest_class_composite_path))[0]
            after_timestamp  = pyeo.filesystem_utilities.get_image_acquisition_time(os.path.basename(class_image_paths[-1]))
            output_product = os.path.join(probability_image_dir,
                                          "report_{}_{}_{}.tif".format(
                                          before_timestamp.strftime("%Y%m%dT%H%M%S"),
                                          tile_id,
                                          after_timestamp.strftime("%Y%m%dT%H%M%S"))
                                          )
            pyeo.raster_manipulation.combine_date_maps(date_image_paths, output_product)
            log.info("Created combined raster file with two layers: {}".format(output_product))

            log.info("Change date layers done and output product aggregated.")

        # ------------------------------------------------------------------------
        # Step 5: Update the baseline composite with the reflectance values of only the changed pixels.
        #         Update last_date of the baseline composite.
        # ------------------------------------------------------------------------

        if do_update or do_all:
            log.info("---------------------------------------------------------------")
            log.info("Updating baseline composite with new imagery.")
            log.info("---------------------------------------------------------------")
            # get all composite file paths
            composite_paths = [ f.path for f in os.scandir(composite_dir) if f.is_file() ]
            if len(composite_paths) == 0:
                raise FileNotFoundError("No composite images found in {}.".format(composite_dir))
            log.info("Sorting composite image list by time stamp.")
            composite_images = \
                pyeo.filesystem_utilities.sort_by_timestamp(
                    [image_name for image_name in os.listdir(composite_dir) if image_name.endswith(".tif")],
                    recent_first=False
                )
            try:
                latest_composite_name = \
                    pyeo.filesystem_utilities.sort_by_timestamp(
                        [image_name for image_name in os.listdir(composite_dir) if image_name.endswith(".tif")],
                        recent_first=True
                    )[0]
                latest_composite_path = os.path.join(composite_dir, latest_composite_name)
                latest_composite_timestamp = pyeo.filesystem_utilities.get_sen_2_image_timestamp(os.path.basename(latest_composite_path))
                log.info("Most recent composite at {}".format(latest_composite_path))
            except IndexError:
                log.critical("Latest composite not found. The first time you run this script, you need to include the "
                             "--build-composite flag to create a base composite to work off. If you have already done this,"
                             "check that the earliest dated image in your images/merged folder is later than the earliest"
                             " dated image in your composite/ folder.")
                sys.exit(1)

            # Find all categorised images
            categorised_paths = [ f.path for f in os.scandir(categorised_image_dir) if f.is_file() ]
            if len(categorised_paths) == 0:
                raise FileNotFoundError("No categorised images found in {}.".format(categorised_image_dir))
            log.info("Sorting categorised image list by time stamp.")
            categorised_images = \
                pyeo.filesystem_utilities.sort_by_timestamp(
                    [image_name for image_name in os.listdir(categorised_image_dir) if image_name.endswith(".tif")],
                    recent_first=False
                )
            # Drop the categorised images that were made before the most recent composite date
            latest_composite_timestamp_datetime = pyeo.filesystem_utilities.get_image_acquisition_time(latest_composite_name)
            categorised_images = [image for image in categorised_images \
                                 if pyeo.filesystem_utilities.get_change_detection_dates(os.path.basename(image))[1] > latest_composite_timestamp_datetime ]

            # Find all L2A images
            l2a_paths = [ f.path for f in os.scandir(l2_masked_image_dir) if f.is_file() ]
            if len(l2a_paths) == 0:
                raise FileNotFoundError("No images found in {}.".format(l2_masked_image_dir))
            log.info("Sorting masked L2A image list by time stamp.")
            l2a_images = \
                pyeo.filesystem_utilities.sort_by_timestamp(
                    [image_name for image_name in os.listdir(l2_masked_image_dir) if image_name.endswith(".tif")],
                    recent_first=False
                )
            
            log.info("Updating most recent composite with new imagery over detected changed areas.")
            for categorised_image in categorised_images:
                # Find corresponding L2A file
                timestamp = pyeo.filesystem_utilities.get_change_detection_date_strings(os.path.basename(categorised_image))
                before_time = timestamp[0]
                after_time = timestamp[1]
                granule = pyeo.filesystem_utilities.get_sen_2_image_tile(os.path.basename(categorised_image))
                l2a_glob = "S2[A|B]_MSIL2A_{}_*_{}_*.tif".format(after_time, granule)
                log.info("Searching for image name pattern: {}".format(l2a_glob))
                l2a_image = glob.glob(os.path.join(l2_masked_image_dir, l2a_glob))
                if len(l2a_image) == 0:
                    log.warning("Matching L2A file not found for categorised image {}".format(categorised_image))
                else:
                    l2a_image = l2a_image[0]
                log.info("Categorised image: {}".format(categorised_image))
                log.info("Matching stacked masked L2A file: {}".format(l2a_image))

                # Extract all reflectance values from the pixels with the class of interest in the classified image
                with TemporaryDirectory(dir=os.getcwd()) as td:
                    log.info("Creating mask file from categorised image {} for class: {}".format(os.path.join(categorised_image_dir, categorised_image), class_of_interest))
                    mask_path = os.path.join(td, categorised_image.split(sep=".")[0]+".msk")
                    log.info("  at {}".format(mask_path))
                    pyeo.raster_manipulation.create_mask_from_class_map(os.path.join(categorised_image_dir, categorised_image), 
                                                                        mask_path, [class_of_interest], buffer_size=0, out_resolution=None) 
                    masked_image_path = os.path.join(td, categorised_image.split(sep=".")[0]+"_change.tif")
                    pyeo.raster_manipulation.apply_mask_to_image(mask_path, l2a_image, masked_image_path)
                    new_composite_path = os.path.join(composite_dir, "composite_{}.tif".format(
                                                      pyeo.filesystem_utilities.get_sen_2_image_timestamp(os.path.basename(l2a_image))))
                    # Update pixel values in the composite over the selected pixel locations where values are not missing
                    log.info("  {}".format(latest_composite_path))
                    log.info("  {}".format([l2a_image]))
                    log.info("  {}".format(new_composite_path))
                    #TODO generate_date_image=True currently produces a type error
                    pyeo.raster_manipulation.update_composite_with_images(
                                                                         latest_composite_path,
                                                                         [masked_image_path], 
                                                                         new_composite_path,  
                                                                         generate_date_image=False,
                                                                         missing=0
                                                                         )
                latest_composite_path = new_composite_path

        # ------------------------------------------------------------------------
        # Step 6: Create quicklooks for fast visualisation and quality assurance of output
        # ------------------------------------------------------------------------

        if do_quicklooks or do_all:
            log.info("---------------------------------------------------------------")
            log.info("Producing quicklooks.")
            log.info("---------------------------------------------------------------")
            dirs_for_quicklooks = [composite_dir, l2_masked_image_dir, categorised_image_dir, probability_image_dir]
            for main_dir in dirs_for_quicklooks: 
                files = [ f.path for f in os.scandir(main_dir) if f.is_file() and os.path.basename(f).endswith(".tif") ] 
                #files = [ f.path for f in os.scandir(main_dir) if f.is_file() and os.path.basename(f).endswith(".tif") and "class" in os.path.basename(f) ] # do classification images only
                if len(files) == 0:
                    log.warning("No images found in {}.".format(main_dir))
                else:    
                    for f in files:
                        log.info("Creating quicklook image from: {}".format(f))
                        quicklook_path = os.path.join(quicklook_dir, os.path.basename(f).split(".")[0]+".png")
                        log.info("                           at: {}".format(quicklook_path))
                        pyeo.raster_manipulation.create_quicklook(f, 
                                                                  quicklook_path,
                                                                  width=512, 
                                                                  height=512, 
                                                                  format="PNG", 
                                                                  bands=[3,2,1],
                                                                  scale_factors=[[0,2000,0,255]]
                                                                  )
            log.info("Quicklooks complete.")


        # ------------------------------------------------------------------------
        # Step 6: Free up disk space by deleting all downloaded Sentinel-2 images and intermediate processing steps
        # ------------------------------------------------------------------------

        # Build new composite
        if do_delete:
            log.info("---------------------------------------------------------------")
            log.info("Deleting downloaded images and intermediate products after use to free up disk space.")
            log.info("---------------------------------------------------------------")
            log.warning("This function is currently disabled.")
            '''
            shutil.rmtree(l1_image_dir)
            shutil.rmtree(l2_image_dir)
            shutil.rmtree(l2_masked_image_dir)
            shutil.rmtree(composite_l1_image_dir)
            shutil.rmtree(composite_l2_image_dir)
            shutil.rmtree(composite_l2_masked_image_dir)
            '''

        # ------------------------------------------------------------------------
        # End of processing
        # ------------------------------------------------------------------------
        log.info("---------------------------------------------------------------")
        log.info("---                  PROCESSING END                         ---")
        log.info("---------------------------------------------------------------")
    
    except Exception:
        log.exception("Fatal error in rolling_s2_composite chain")


if __name__ == "__main__":

    # Reading in config file
    parser = argparse.ArgumentParser(description='Downloads, preprocesses and classifies Sentinel 2 images. A directory'
                                                 'structure to contain preprocessed and downloaded files will be'
                                                 'created at the root_dir location specified in the config file.'
                                                 'If any of the step flags are present, only those '
                                                 'steps will be performed - otherwise all processing steps will be '
                                                 'performed.')
    parser.add_argument(dest='config_path', action='store', default=r'change_detection.ini',
                        help="A path to a .ini file containing the specification for the job. See "
                             "pyeo/apps/change_detection/change_detection.ini for an example.")
    parser.add_argument('--start_date', dest='arg_start_date', help="Overrides the start date in the config file. Set to "
                                                                "LATEST to get the date of the last merged accquistion")
    parser.add_argument('--end_date', dest='arg_end_date', help="Overrides the end date in the config file. Set to TODAY"
                                                            "to get today's date")
    parser.add_argument('-b', '--build_composite', dest='build_composite', action='store_true', default=False,
                        help="If present, creates a cloud-free (ish) composite between the two dates specified in the "
                             "config file.")
    parser.add_argument("--tile", dest="tile_id", type=str, default="None", help="Overrides the geojson location with a"
                                                                                  "Sentinel-2 tile ID location")
    parser.add_argument("--chunks", dest="chunks", type=int, default=10, help="Sets the number of chunks to split "
                                                                                  "images to in ml processing")
    parser.add_argument('-d', '--download', dest='do_download', action='store_true', default=False,
                        help='If present, perform the query and download level 1 images.')
    parser.add_argument('--download_source', default="scihub", help="Sets the download source, can be scihub "
                                                                    "(default) or aws")
    parser.add_argument('-c', '--classify', dest='do_classify', action='store_true', default=False,
                        help="For each image in images/stacked, applies the classifier given in the .ini file. Saves"
                        "the outputs in output/categories.")
    parser.add_argument('-p', '--build_prob_image', dest='build_prob_image', action='store_true', default=False,
                        help="If present, build a confidence map of pixels. These tend to be large.")
    parser.add_argument('-u', '--update', dest='do_update', action='store_true', default=False,
                        help="Builds a new cloud-free composite in composite/ from the latest image and mask"
                             " in images/merged")
    parser.add_argument('-q', '--quicklooks', dest='do_quicklooks', action='store_true', default=False,
                        help="Creates quicklooks for all composites, L2A change images, classified images and probability images.")
    parser.add_argument('-r', '--remove', dest='do_delete', action='store_true', default=False,
                        help="Not implemented. If present, removes all images in images/ to save space.")

    args = parser.parse_args()

    # TODO: bands and resolution can be made flexible BUT the bands need to be at the same resolution
    bands = ['B02', 'B03', 'B04', 'B08']
    resolution = '10m'
    buffer_size = 20            #set buffer in number of pixels for dilating the SCL cloud mask (recommend 30 pixels of 10 m) for the change detection
    buffer_size_composite = 10  #set buffer in number of pixels for dilating the SCL cloud mask (recommend 10 pixels of 10 m) for the composite building
    max_image_number = 30       #maximum number of images to be downloaded for compositing, in order of least cloud cover
    from_classes = [1]          #find subsequent changes from any of these classes
    to_classes = [2,3,4,5,7,11] #                          to any of these classes
    skip_existing = True        # skip existing image products from processing
    faulty_granule_threshold = 400 # granules below this size in MB will not be downloaded

    '''
    e.g. classes in new matogrosso model	
    1	primary forest
    2	plantation forest
    3	bare soil
    4	crops
    5	grassland
    6	open water
    7	burn scar
    8	cloud
    9	cloud shadow
    10	haze
    11	open woodland
    '''


    rolling_detection(**vars(args))
