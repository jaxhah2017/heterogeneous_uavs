class HetergeneousUav(object):
    def __init__(self, n_uav: int, n_types: int = 2) -> None:
        super(HetergeneousUav, self).__init__()
        
        self.service_capacity = [3, 5, 7]
        self.coverage_capacity = [50, 70, 100]
        self.communication_range = [50, 70, 100]

        self.n_types = n_types

    def get_uav_type(self, type):
        pass
        # return dict(sv_capacity=self.service_capacity[type], 
        #             cv_capacity=self.coverage_capacity[type], 
        #             comm_capacity=self.coverage_capacity[type])

    
        
    