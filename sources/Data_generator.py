from Doodler import *
import random

class Data_generator():

    # Create se set of parameters for the data generator
    def __init__(self,number_data=100, n=25, noise=0, tr_size=0.7, test_size=0.2, wr=[0.3,0.7], hr=[0.3,0.7], flat=False,random_seed = None) -> None:

        # Verify that the parameters are valid
        self.validate_parameter("number_data", number_data, 1, 1000)
        self.validate_parameter("n", n, 10, 50)
        self.validate_parameter("noise", noise, 0, 1)
        self.validate_parameter("tr_size", tr_size, 0, 1)
        self.validate_parameter("test_size", test_size, 0, 1)
        self.validate_parameter("tr_size+test_size", tr_size + test_size, 0, 1, error_message="tr_size+test_size must be less than 1")
        self.validate_parameter("wr", wr[0], 0, 1,error_message="wr[0] must be less than between 0 and 1")
        self.validate_parameter("wr", wr[1], 0, 1,error_message="wr[1] must be less than between 0 and 1")
        self.validate_parameter("wr", wr[0], 0, wr[1],error_message="wr[0] must be less than wr[1]")
        self.validate_parameter("hr", hr[0], 0, 1,error_message="hr[0] must be less than between 0 and 1")
        self.validate_parameter("hr", hr[1], 0, 1,error_message="hr[1] must be less than between 0 and 1")
        self.validate_parameter("hr", hr[0], 0, hr[1],error_message="hr[0] must be less than hr[1]")
        self.validate_boolean("flat", flat)

        # Set the parameters
        self.number_data = number_data
        self.n = n
        self.noise = noise
        self.tr_size = tr_size
        self.test_size = test_size
        self.val_size = 1 - tr_size - test_size
        self.wr = wr
        self.hr = hr
        self.flat = flat
        self.random_seed = random_seed

        # Data variables
        self.tr_data = None
        self.test_data = None
        self.val_data = None

    # Fonction to raise an error if the parameters are not valid (range issues)
    def validate_parameter(self, name, value, min_value, max_value, error_message=None):
        if not min_value <= value <= max_value:
            error_message = error_message or f"{name} must be between {min_value} and {max_value}"
            raise ValueError(error_message)

    # Fonction to raise an error if the parameters are not valid (type issues)
    def validate_boolean(self, name, value):
        if type(value) != bool:
            raise TypeError(f"{name} must be a boolean")

    
    # Split the data into 3 sets
    def split_data(self,data):

        # Create a list from the range
        data_range = list(range(self.number_data))
        
        # Shuffle the list randomly
        random.shuffle(data_range)
        
        # Calculate the sizes for each set
        tr_end = int(self.tr_size * self.number_data)
        test_end = int((self.tr_size + self.test_size) * self.number_data)

        # Split the list into 3 sets
        tr_index = data_range[:tr_end]
        test_index = data_range[tr_end:test_end]
        val_index = data_range[test_end:]

        # create each set of data
        tr_data = [data[0][i] for i in tr_index]
        test_data = [data[0][i] for i in test_index]
        val_data = [data[0][i] for i in val_index]

        # target data
        tr_target = [data[1][i] for i in tr_index]
        test_target = [data[1][i] for i in test_index]
        val_target =  [data[1][i] for i in val_index]

        # labels
        tr_label = [data[2][i] for i in tr_index]
        test_label = [data[2][i] for i in test_index]
        val_label = [data[2][i] for i in val_index]

        # make the effective split
        self.tr_data = (tr_data, tr_target, tr_label, data[3], data[4])
        self.test_data = (test_data, test_target, test_label, data[3], data[4])
        self.val_data = (val_data, val_target, val_label, data[3], data[4])
        
        

    # Generate the data
    def generate_data(self,return_data=False):
        
        if self.random_seed is not None:
            random.seed(self.random_seed)
            
        # Generate the data
        data = gen_standard_cases(  count=self.number_data,rows=self.n,cols=self.n,wr=self.wr,hr=self.hr,
                                    noise=self.noise, cent=False, show=False, flat=self.flat,
                                    fc=(1,1),auto=False,mono=True,one_hots=True,multi=False)
        # Split the data
        self.split_data(data)

        # Return the data if needed
        if return_data:
            return self.get_data()

        print("Data generated")

    # Return the data
    def get_data(self):
        return self.tr_data, self.test_data, self.val_data
    
    # Save Data 
    def save_data(self, path):
        dump_doodle_cases(self.tr_data,path+"/tr_data")
        dump_doodle_cases(self.test_data,path+"/test_data")
        dump_doodle_cases(self.val_data,path+"/val_data")
        print("Data saved at : ",path)

# Load Data
def load_data(path):
    tr_data = load_doodle_cases(path+"/tr_data")
    test_data = load_doodle_cases(path+"/test_data")
    val_data = load_doodle_cases(path+"/val_data")
    print("Data loaded from : ",path)
    return tr_data, test_data, val_data