import pyeo.classification
import pyeo.filesystem_utilities
import configparser
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Produces a trained random forest model from a folder containing '
                                                 'rasters and shapefiles')
    parser.add_argument("region_path", type=str, action='store',
                        help="Path to the folder containing the region data")
    parser.add_argument("class_attribute_field", type=str, action='store',
                        help="Attribute field holding the class to train on")
    parser.add_argument("out_path", type=str, action="store",
                        help="Path for the output .pkl file")
    args = parser.parse_args()

    conf = configparser.ConfigParser()

    log = pyeo.filesystem_utilities.init_log("model.log")

    log.info("***MODEL CREATION START***")
    log.info("Region path: {}".format(args.region_path))
    log.info("Output path for the model: {}".format(args.out_path))
    log.info("Attribute field in the shapefile with training labels: {}".format(args.class_attribute_field))

    pyeo.classification.create_rf_model_for_region(args.region_path, args.out_path, args.class_attribute_field)

    log.info("***MODEL CREATION END***")
