'''
BMI code for the HBV (Hydrologiska byrÂns vattenavdelning) model.

This BMI code has been developed by Motasem S Abualqumboz, Utah State University (2022).
The code was developed using the BMI code developed by Jonathan Frame for the 
CFE (Conceptual Fuctional Eqvalent) model (https://github.com/jmframe/si_2022_train).

'''

#---------- Python libraries ----------#

from pathlib import Path

# Basic utilities
import numpy as np
import pandas as pd
import yaml # Configuration file functionality
import hbv # The model we want to run

from bmipy import Bmi # Need these for BMI

#---------- BMI class for the HBV model ----------#


class hbv_bmi(Bmi):
    def __init__(self):
        """Create a model that is ready for initialization."""
        super(hbv_bmi, self).__init__()
        self._values = {}
        self._var_loc = "node"
        self._var_grid_id = 0
        self._start_time = 0.0
        self._end_time = np.finfo(float).max
        self.var_array_lengths = 1       
        
    #---------------------------------------------
    # Input variable names (CSDMS standard names)
    #---------------------------------------------
    _input_var_names = ['timestep_rainfall_input', 'temperature_k']

    #---------------------------------------------
    # Output variable names (CSDMS standard names)
    #---------------------------------------------
    _output_var_names = ['water_output']

    #------------------------------------------------------
    # Create a Python dictionary that maps CSDMS Standard
    # Names to the model's internal variable names.
    # This is going to get long, 
    #     since the input variable names could come from any forcing...
    #------------------------------------------------------
    #_var_name_map_long_first = {
    _var_units_map = {  'timestep_rainfall_input':'m/h',
                        'temperature_k':'K',
                        'water_output':'m/h'
                        }

    #------------------------------------------------------
    # A list of static attributes/parameters.
    #------------------------------------------------------
    _model_parameters_list = ["latitude",
                              "angular_velocity",
                              "threshold_temperature_TT",
                              "snowfall_correction_factor_SFCF",
                              "snow_melt_degreeDay_factor_CFMAX",
                              "Water_holding_capacity_CWH",
                              "refreezing_coefficient_CFR",
                              "field_capacity_FC",
                              "evaporation_reduction_threshold_LP",
                              "shape_coefficient_beta",
                              "recession_constant_near_surface_K0",
                              "recession_constant_upper_storage_K1",
                              "recession_constant_lower_storage_K2",
                              "threshold_for_shallow_storage_UZL",
                              "lower_to_upper_maxflow_Percolation",
                              "MAXBAS"]

    #------------------------------------------------------------
    #------------------------------------------------------------
    # BMI: Model Control Functions
    #------------------------------------------------------------ 
    #------------------------------------------------------------

    #-------------------------------------------------------------------
    def initialize(self, bmi_cfg_file_name: str ):
        
        # -------------- A default value if left unconfigured ------------------#
        # self._values["angular_velocity"] = 0.5
                
        # -------------- Read in the BMI configuration -------------------------#
        if not isinstance(bmi_cfg_file_name, str) or len(bmi_cfg_file_name) == 0:
            raise RuntimeError("No BMI initialize configuration provided, nothing to do...")
            
        # Get the absolute path
        bmi_cfg_file = Path(bmi_cfg_file_name).resolve()
        
        if not bmi_cfg_file.is_file():
            raise RuntimeError("No configuration provided, nothing to do...")

        with bmi_cfg_file.open('r') as fp:
            cfg = yaml.safe_load(fp)
            
        #declares the location of the configuration setting to be tested, and then what value is to be tested
        
        self.cfg_bmi = self._parse_config(cfg) 

        # ------------- Initialize the parameters, inputs and outputs ----------#
        for parm in self._model_parameters_list:
            self._values[parm] = self.cfg_bmi[parm]
        for model_input in self.get_input_var_names():
            self._values[model_input] = np.zeros(1, dtype=float)
        for model_output in self.get_output_var_names():
            if model_output == "water_output":
                self._values["water_output"] = np.zeros(1, dtype=float)
            else:
                pass

        # ------------- Set time to initial value -----------------------#
        self._values['current_model_time'] = self.cfg_bmi['initial_time']
        
        # ------------- Set time step size -----------------------#
        self._values['time_step_seconds'] = self.cfg_bmi['time_step_seconds']
        
        # -------------- A default value if left unconfigured ------------------#
        
        # self._values["angular_velocity"] = 0.5
        # self._values['latitude'] = self.cfg_bmi['latitude']
        # self._values['angular_velocity'] = self.cfg_bmi['angular_velocity']
        # self._values['threshold_temperature_TT'] = self.cfg_bmi['threshold_temperature_TT']
        # self._values['snowfall_correction_factor_SFCF'] = self.cfg_bmi['snowfall_correction_factor_SFCF']
        # self._values['snow_melt_degreeDay_factor_CFMAX'] = self.cfg_bmi['snow_melt_degreeDay_factor_CFMAX']
        # self._values['Water_holding_capacity_CWH'] = self.cfg_bmi['Water_holding_capacity_CWH']
        # self._values['refreezing_coefficient_CFR'] = self.cfg_bmi['refreezing_coefficient_CFR']
        # self._values['field_capacity_FC'] = self.cfg_bmi['field_capacity_FC']
        # self._values['evaporation_reduction_threshold_LP'] = self.cfg_bmi['evaporation_reduction_threshold_LP']
        # self._values['shape_coefficient_beta'] = self.cfg_bmi['shape_coefficient_beta']
        # self._values['recession_constant_near_surface_K0'] = self.cfg_bmi['recession_constant_near_surface_K0']
        # self._values['recession_constant_upper_storage_K1'] = self.cfg_bmi['recession_constant_upper_storage_K1']
        # self._values['recession_constant_lower_storage_K2'] = self.cfg_bmi['recession_constant_lower_storage_K2']
        # self._values['threshold_for_shallow_storage_UZL'] = self.cfg_bmi['threshold_for_shallow_storage_UZL']
        # self._values['lower_to_upper_maxflow_Percolation'] = self.cfg_bmi['lower_to_upper_maxflow_Percolation']
        # self._values['MAXBAS'] = self.cfg_bmi['MAXBAS']
        
        self.latitude = self.cfg_bmi['latitude']
        self.angular_velocity = self.cfg_bmi['angular_velocity']
        self.threshold_temperature_TT = self.cfg_bmi['threshold_temperature_TT']
        self.snowfall_correction_factor_SFCF = self.cfg_bmi['snowfall_correction_factor_SFCF']
        self.snow_melt_degreeDay_factor_CFMAX = self.cfg_bmi['snow_melt_degreeDay_factor_CFMAX']
        self.Water_holding_capacity_CWH = self.cfg_bmi['Water_holding_capacity_CWH']
        self.refreezing_coefficient_CFR = self.cfg_bmi['refreezing_coefficient_CFR']
        self.field_capacity_FC = self.cfg_bmi['field_capacity_FC']
        self.evaporation_reduction_threshold_LP = self.cfg_bmi['evaporation_reduction_threshold_LP']
        self.shape_coefficient_beta = self.cfg_bmi['shape_coefficient_beta']
        self.recession_constant_near_surface_K0 = self.cfg_bmi['recession_constant_near_surface_K0']
        self.recession_constant_upper_storage_K1 = self.cfg_bmi['recession_constant_upper_storage_K1']
        self.recession_constant_lower_storage_K2 = self.cfg_bmi['recession_constant_lower_storage_K2']
        self.threshold_for_shallow_storage_UZL = self.cfg_bmi['threshold_for_shallow_storage_UZL']
        self.lower_to_upper_maxflow_Percolation = self.cfg_bmi['lower_to_upper_maxflow_Percolation']
        self.MAXBAS = self.cfg_bmi['MAXBAS']
        
        # output
        self.unrouted_streamflow_through_channel_network_Qgen = 0.0
        
        #input
        self.timestep_rainfall_input_m = 0.0
        self.temperature = 0.0
        self.average_watershed_potential_et = 0.0
        
        # Snow routine
        self.simulated_snowfall_SF = 0.0
        self.catchment_input_inc = 0.0
        self.snowpack_melting_rate_melt  = 0.0
        self.refreeze = 0.0
        
        # soil routine
        self.average_watershed_actual_aet = 0.0
        self.soil=0.0
        self.old_soil_storage_content_oldSM = 0.0
        self.y = 0.0
        self.m = 0.0
        self.partitioning_function_dQdP = 0.0
        self.mean_storage_content_meanSM = 0.0
        self.recharge = 0.0
        
        # Reservoir
        self.shallow_flow_Qstz = 0.0
        self.flow_from_upper_storage_Qsuz = 0.0
        self.flow_from_lower_storage_Qslz = 0.0
        self.storage_from_upper_GW_reservoir_S1 = 0.0
        self.storage_from_lower_GW_reservoir_S2 = 0.0
        self.total_storage_Storage = 0.0
               
        # Model state variables
        self.snow_water_equivalent_SWE = 0.0                      # Initial snow water equivalent
        self.upper_zone_storage_SUZ = 0.0                         # Initial upper zone storage
        self.lower_zone_storage_SLZ = 0.0                         # Initial lower zone storage
        self.simulated_snowpack_SP = 152.4                        # Initial value for simulated snowpack
        self.liquid_water_in_snowpack_WC = 0.0                    # Initial liquid water in snowpack
        self.soil_storage_content_SM = self._values["field_capacity_FC"]     # Initial soil storage content

        
        
        #############################################################################
        # _________________________________________________________________________ #
        # _________________________________________________________________________ #
        # CREATE AN INSTANCE OF THE Hydrologiska Byråns Vattenbalansavdelning (HBV) #
        self.hbv_model = hbv.HBV()
        # _________________________________________________________________________ #
        # _________________________________________________________________________ #
        #############################################################################


    #------------------------------------------------------------ 
    def update(self):
        """
        Update/advance the model by one time step.
        """
        model_time_after_next_time_step = self._values['current_model_time'] + self._values['time_step_seconds']
        self.update_until(model_time_after_next_time_step)
    
    #------------------------------------------------------------ 
    def update_until(self, future_time: float):
        """
        Update the model to a particular time

        Parameters
        ----------
        future_time : float
            The future time to when the model should be advanced.
        """
        update_delta_t = future_time - self._values['current_model_time'] 

        # Update HBV model inputs
        self.timestep_rainfall_input_m = self._values["timestep_rainfall_input"]
        self.temperature = self._values["temperature_k"]
        self.average_watershed_potential_et = 0.0
        
        # Model Update
        self.hbv_model.run_hbv(self)
        
        # Model Output
        # print('Model-streamflow',self.unrouted_streamflow_through_channel_network_Qgen)
        self._values["water_output"][0] = self.unrouted_streamflow_through_channel_network_Qgen
        
        # print('Model-streamflow2', self._values["water_output"][0],'\n')

        # Update our clock
        self._values['current_model_time'] = self._values['current_model_time'] + update_delta_t

    #------------------------------------------------------------    
    def finalize( self ):
        """Finalize model."""
        pass

    #--------------------------------------------------------
    # Note: These are currently variables needed from other
    #       components vs. those read from files or GUI.
    #--------------------------------------------------------   
    def get_input_var_names(self):

        return self._input_var_names

    def get_output_var_names(self):
 
        return self._output_var_names

    #------------------------------------------------------------ 
    def get_component_name(self):
        """Name of the component."""
        return "BMI Toy Model"

    #------------------------------------------------------------ 
    def get_input_item_count(self):
        """Get names of input variables."""
        return len(self._input_var_names)

    #------------------------------------------------------------ 
    def get_output_item_count(self):
        """Get names of output variables."""
        return len(self._output_var_names)

    #------------------------------------------------------------ 
    def get_value(self, var_name: str, dest: np.ndarray) -> np.ndarray:
        """Copy of values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : ndarray
            A numpy array into which to place the values.
        Returns
        -------
        array_like
            Copy of values.
        """
        dest[:] = self.get_value_ptr(var_name)
        return dest

    #     #------------------------------------------------------------ 
    # def get_value(self, var_name):
    #     """Copy of values.
    #     Parameters
    #     ----------
    #     var_name : str
    #         Name of variable as CSDMS Standard Name.
    #     dest : ndarray
    #         A numpy array into which to place the values.
    #     Returns
    #     -------
    #     array_like
    #         Copy of values.
    #     """
    #     return self.get_value_ptr(var_name)

    #-------------------------------------------------------------------
    def get_value_ptr(self, var_name: str) -> np.ndarray:
        """Reference to values.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        array_like
            Value array.
        """
        
        return self._values[var_name]

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # BMI: Variable Information Functions
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    def get_var_name(self, long_var_name):
                              
        return self._var_name_map_long_first[ long_var_name ]

    #-------------------------------------------------------------------
    def get_var_units(self, long_var_name):

        return self._var_units_map[ long_var_name ]
                                                             
    #-------------------------------------------------------------------
    def get_var_type(self, var_name: str) -> str:
        """Data type of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Data type.
        """
        return str(self.get_value_ptr(var_name).dtype)
    
    #------------------------------------------------------------ 
    def get_var_grid(self, name):
        
        # all vars have grid 0 but check if its in names list first
        if name in (self._output_var_names + self._input_var_names):
            return self._var_grid_id  

    #------------------------------------------------------------ 
    def get_var_itemsize(self, name):
        return self.get_value_ptr(name).itemsize

    #------------------------------------------------------------ 
    def get_var_location(self, name):
        
        if name in (self._output_var_names + self._input_var_names):
            return self._var_loc

    #-------------------------------------------------------------------
    def get_var_rank(self, long_var_name):

        return np.int16(0)

    #-------------------------------------------------------------------
    def get_start_time( self ):
    
        return self._start_time 

    #-------------------------------------------------------------------
    def get_end_time( self ) -> float:

        return self._end_time 


    #-------------------------------------------------------------------
    def get_current_time( self ):

        return self._values['current_model_time']

    #-------------------------------------------------------------------
    def get_time_step( self ):

        return self._values['time_step_seconds']

    #-------------------------------------------------------------------
    def get_time_units( self ):

        return "seconds" 
       
    #-------------------------------------------------------------------
    def set_value(self, var_name, values: np.ndarray):
        """
        Set model values for the provided BMI variable.

        Parameters
        ----------
        var_name : str
            Name of model variable for which to set values.
        values : np.ndarray
              Array of new values.
        """ 
        self._values[var_name][:] = values

    #------------------------------------------------------------ 
    def set_value_at_indices(self, var_name: str, indices: np.ndarray, src: np.ndarray):
        """
        Set model values for the provided BMI variable at particular indices.

        Parameters
        ----------
        var_name : str
            Name of model variable for which to set values.
        indices : array_like
            Array of indices of the variable into which analogous provided values should be set.
        src : array_like
            Array of new values.
        """
        # This is not particularly efficient, but it is functionally correct.
        for i in range(indices.shape[0]):
            bmi_var_value_index = indices[i]
            self.get_value_ptr(var_name)[bmi_var_value_index] = src[i]

    #------------------------------------------------------------ 
    def get_var_nbytes(self, var_name) -> int:
        """
        Get the number of bytes required for a variable.
        Parameters
        ----------
        var_name : str
            Name of variable.
        Returns
        -------
        int
            Size of data array in bytes.
        """
        return self.get_value_ptr(var_name).nbytes

    #------------------------------------------------------------ 
    def get_value_at_indices(self, var_name: str, dest: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Get values at particular indices.
        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : np.ndarray
            A numpy array into which to place the values.
        indices : np.ndarray
            Array of indices.
        Returns
        -------
        np.ndarray
            Values at indices.
        """
        original: np.ndarray = self.get_value_ptr(var_name)
        for i in range(indices.shape[0]):
            value_index = indices[i]
            dest[i] = original[value_index]
        return dest

    # JG Note: remaining grid funcs do not apply for type 'scalar'
    #   Yet all functions in the BMI must be implemented 
    #   See https://bmi.readthedocs.io/en/latest/bmi.best_practices.html          
    #------------------------------------------------------------ 
    def get_grid_edge_count(self, grid):
        raise NotImplementedError("get_grid_edge_count")

    #------------------------------------------------------------ 
    def get_grid_edge_nodes(self, grid, edge_nodes):
        raise NotImplementedError("get_grid_edge_nodes")

    #------------------------------------------------------------ 
    def get_grid_face_count(self, grid):
        raise NotImplementedError("get_grid_face_count")
    
    #------------------------------------------------------------ 
    def get_grid_face_edges(self, grid, face_edges):
        raise NotImplementedError("get_grid_face_edges")

    #------------------------------------------------------------ 
    def get_grid_face_nodes(self, grid, face_nodes):
        raise NotImplementedError("get_grid_face_nodes")
    
    #------------------------------------------------------------
    def get_grid_node_count(self, grid):
        raise NotImplementedError("get_grid_node_count")

    #------------------------------------------------------------ 
    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        raise NotImplementedError("get_grid_nodes_per_face") 
    
    #------------------------------------------------------------ 
    def get_grid_origin(self, grid_id, origin):
        raise NotImplementedError("get_grid_origin") 

    #------------------------------------------------------------ 
    def get_grid_rank(self, grid_id):
        if grid_id == 0: 
            return 1

    #------------------------------------------------------------ 
    def get_grid_shape(self, grid_id, shape):
        raise NotImplementedError("get_grid_shape") 

    #------------------------------------------------------------ 
    def get_grid_size(self, grid_id):
        if grid_id == 0:
            return 1

    #------------------------------------------------------------ 
    def get_grid_spacing(self, grid_id, spacing):
        raise NotImplementedError("get_grid_spacing") 

    #------------------------------------------------------------ 
    def get_grid_type(self, grid_id=0):
        if grid_id == 0:
            return 'scalar'

    #------------------------------------------------------------ 
    def get_grid_x(self):
        raise NotImplementedError("get_grid_x") 

    #------------------------------------------------------------ 
    def get_grid_y(self):
        raise NotImplementedError("get_grid_y") 

    #------------------------------------------------------------ 
    def get_grid_z(self):
        raise NotImplementedError("get_grid_z") 


    #------------------------------------------------------------ 
    #------------------------------------------------------------ 
    #-- Random utility functions
    #------------------------------------------------------------ 
    #------------------------------------------------------------ 

    def _parse_config(self, cfg):
        for key, val in cfg.items():
            # convert all path strings to PosixPath objects
            if any([key.endswith(x) for x in ['_dir', '_path', '_file', '_files']]):
                if (val is not None) and (val != "None"):
                    if isinstance(val, list):
                        temp_list = []
                        for element in val:
                            temp_list.append(Path(element))
                        cfg[key] = temp_list
                    else:
                        cfg[key] = Path(val)
                else:
                    cfg[key] = None

            # convert Dates to pandas Datetime indexs
            elif key.endswith('_date'):
                if isinstance(val, list):
                    temp_list = []
                    for elem in val:
                        temp_list.append(pd.to_datetime(elem, format='%d/%m/%Y'))
                    cfg[key] = temp_list
                else:
                    cfg[key] = pd.to_datetime(val, format='%d/%m/%Y')

            else:
                pass

        # Add more config parsing if necessary
        return cfg