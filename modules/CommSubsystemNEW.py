import numpy as np
import math



class CommSubsystem():
    #""
        #Implementation of communication subsystem.
        #The data are taken from commercially available components.
    #"""

    def __init__(self):
         # self.POWER = 10 #[W]
        # self.RXGAIN = 30 #[dB]
        # self.RXLOSS = 1 #[dB]
        # self.TXGAIN = 30 #[dB]
        # self.TXLOSS = 3 #[dB]
        # self.FREQUENCY = 4e9 #[Hz]
        # self.BANDWIDTH = 1e6 #[Hz]
        # self.SYMBOLRATE = 1e3 #[Hz]
        # self.MODULATIONORDER = 4 #QPSK
        # self.SENSITIVITY = -115 #[dBW]
        # # other attributes ...


        self.POWER = 2 #[W]
        self.RXGAIN = 1 #[dB]
        self.RXLOSS = 0.5 #[dB]
        self.TXGAIN = 1 #[dB]
        self.TXLOSS = 3 #[dB]
        self.FREQUENCY = 437e6 #[Hz]
        self.BANDWIDTH = 9600 #[Hz]
        self.SYMBOLRATE = 9600 #[Hz]
        self.MODULATIONORDER = 4  #QPSK
        self.SENSITIVITY = -151 #[dBW]
        #self.AVAILABLEENERGY = 5000 #[J]
  #      self.current_data = initial_data

    #def remove_data(self, data):
     #   """ Removes data from sorage """
      #  self.current_data = self.current_data - data

       # if self.current_data < 0:
        #    self.current_data = 0

   # def reset_data_storage(self, initial_data): #Joule
    #    """ Resets current_data to intial value """
     #   self.current_data = initial_data
    
        
    
    #'''
    #    CALCULATION FUNCTIONS
    #'''
    
    def calculateFreeSpaceLoss(self, dist):
        fsllinear = (4 * np.pi * dist * self.FREQUENCY / 3e8)**2
        return 10*np.log10(fsllinear)
    
    def calculateNoise(self):
        return 10*np.log10(1.38e-23 * 290 * self.BANDWIDTH) #kTB assuming temperature of 290K (17C), if we want to include other losses, we have to use 700-1000 K 
    
    def calculateSNR(self, dist):
        return (10*np.log10(self.POWER) + self.RXGAIN + self.TXGAIN - self.RXLOSS - self.TXLOSS - self.calculateFreeSpaceLoss(dist)) - self.calculateNoise()
    
    def calculateIdealDataRate(self, dist):
        # This is the Shannon-Hartley Theorem
        IdealDataRate=self.BANDWIDTH * np.log2(1 + 10**(self.calculateSNR(dist)/10)) #bits/s
        
        if IdealDataRate> self.SYMBOLRATE * np.log2(self.MODULATIONORDER): #In the case of which the channel is limiting my data rate
            return self.SYMBOLRATE * np.log2(self.MODULATIONORDER) 
        else:
            return IdealDataRate
    
    def calculateBER(self, dist):

        bitrate = self.SYMBOLRATE * np.log2(self.MODULATIONORDER)
        efficiency = bitrate / self.BANDWIDTH
        ebn0 = 10**(self.calculateSNR(dist)/10) / efficiency
        return (1/np.log2(self.MODULATIONORDER)) * math.erfc(np.sqrt(2*ebn0))
    
    def calculateEffectiveDataRate(self, dist):

        '''
            This function calculates the effective data rate based on the distance and other parameters.
            The effective data rate is the maximum data rate that can be achieved given the current conditions.
            
            Parameters:
            dist (float): The distance between the transmitter and receiver in meters.
            
            Returns:
            float: The effective data rate in bits per second.
        '''

        receivedPower = (10*np.log10(self.POWER) + self.RXGAIN + self.TXGAIN - self.RXLOSS - self.TXLOSS - self.calculateFreeSpaceLoss(dist))
        if (receivedPower >= self.SENSITIVITY):
            return self.calculateIdealDataRate(dist) * (1 - self.calculateBER(dist)) #bits/sec
        else:
            return 0
    
#    '''
#        PARAMETER UPDATE FUNCTIONS
#    '''
#    def DataSent(self, dist,dt):
#        return self.calculateEffectiveDataRate(dist) * dt
    
#    def DataToSend(self, data_sent):
#        self.current_data = self.current_data - data_sent
#        if self.current_data < 0:
#            self.current_data = 0
# # Instantiate the class
# comm = CommSubsystem()

# # Define a value for 'dist'
# dist = 2400000
# # Call the method on the instance with the defined 'dist'
# testdist = comm.calculateEffectiveDataRate(dist)

# # Print the result
# print(testdist)