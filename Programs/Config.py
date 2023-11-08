import yaml
import warnings

class Config:
    """
        Class that obtains all the variables from the configuration file
    """
    #   Default values

    # General Variables
    imageSizeY = 640
    imageSizeX = 640
    averageSizeOfFish = 70
    randomizeFish = 1
    dataDirectory = 'data'
    amountOfData = 50000
    fractionForTraining = .9
    shouldSaveImages = True
    shouldSaveAnnotations = True

    # Noise Variables
    shouldAddPatchyNoise = True
    shouldAddStaticNoise = True
    averageAmountOfPatchyNoise = .2

    #   Variable relating to the distribution of the fish
    maxFishesInView = 12
    # The following variable is used as the lambda value for a poisson distribution
    averageFishInEdges = 3
    overlappingFishFrequency = .5
    # The following variable is the minimum distance 2 overlapping fishes will be
    maxOverlappingOffset = 10

    #   Thresholds
    # This threshold is used when sequential keypoints of a fish have an x or y value that are about the same
    # and stops it from generating a box that captures that part of the fish
    minimumSizeOfBox = 3
    # The following variable is similar to the one above except that it is for the bounding box of the fish passed
    # to Yolo.  This is necessary because sometimes the fish are barely visible at the edge causing the model to
    # learn to detect the edges as fish
    boundingBoxThreshold = 2
    # Value in which the brightness of a fish is considered solid
    visibilityThreshold = 25

    # None for now since it is going to get set after checking the yaml file
    biggestIdx4TrainingData = None

    # TODO: try setting this to a static method to make it more natural
    def __init__(self, pathToYamlFile):
        """
            Essentially just a function to update the variables accordingly
        """
        static_vars = list(vars(Config))[2:-3]

        file = open(pathToYamlFile, 'r')
        config = yaml.safe_load(file)
        keys = config.keys()
        list_of_vars_in_config = list(keys)

        # Updating the static variables
        for var in list_of_vars_in_config:
            if var in static_vars:
                value = config[var]
                line = 'Config.' + var + ' = '

                if not isinstance(value, str):
                    line += str(value)
                else:
                    line += "'" + value + "'"
                exec(line)
            else:
                warnings.warn(var + ' is not a valid variable, could be a spelling issue')

        Config.biggestIdx4TrainingData = Config.amountOfData * Config.fractionForTraining
        Config.dataDirectory += '/'

        # NOTE: the following was just left as an example for now
        # # Writing the variables to the corresponding classes static variables
        # Config.set_aquarium_vars()
        # Config.set_bounding_box_vars()
    # @staticmethod
    # def set_bounding_box_vars():
    #     print('setting the bounding box vars')

# Setting the variables of the Configuration Class
Config('Inputs/config.yaml')