import numpy as np
import matplotlib.pyplot as plt


class OpticPayload: 

    def __init__(self):
        
        self.DIAMETER=0.09 #Diameter of the optic to fit in a Cubesat'
        self.WAVELENGTH = 700e-9 #Upper limit of visible light
        #self.dim_obj_list= [0.3, 3, 5, 10, 25]
        self.dim_obj= 10 #Size of the object to be detected (100m for ISS)
        # labels = ['6U Cubesat', 'Iridium', 'Starlink', 'Sentinel', 'Envisat']

    

    def diffraction_limit(self, distance):
        """
        Calculate the diffraction limit for an optical system.
        
        Parameters:
        - diameter: aperture diameter (in meters)
        - distances: list of distances to consider
        - wavelength: light wavelength (in meters, default is 700 nm)
        
        Returns:
        - diffraction angle (in radians)
        """
        return 1.22 * self.WAVELENGTH * distance / self.DIAMETER 

    def dist_detect(self):
        """
        Calculate the minimum distance at which objects of various sizes can be resolved.
        
        Parameters:
        - diameter: aperture diameter (in meters)
        - dim_obj_list: list of object sizes (in meters)
        - wavelength: light wavelength (in meters, default is 700 nm)
        
        Returns:
        - list of minimum resolvable distances (in meters) for each object size
        """
        #dim_obj_array = np.array(self.dim_obj_list)  # Convert to numpy array
        #howfar = dim_obj_array * self.DIAMETER / (2.44 * self.WAVELENGTH)
        
        howfar = self.dim_obj* self.DIAMETER / (1.22 * self.WAVELENGTH)

        return howfar
    
    def possibility_observation(self,distance):

        #This allows to confirm if an object can be detected
        
        if (self.dim_obj >= self.diffraction_limit(distance)):
            return 1                    	            #THIS HAS TO BE CHANGED 
        else:
            return 0





# # Define the aperture diameters
# D = [5e-2, 10e-2, 15e-2, 20e-2]
#D = 0.09  #Diameter of the optic to fit in a Cubesat


# # Distances to consider (from 1km to 1000km)
# distances = np.linspace(1e3, 1e6, 500)  # in meters

# # Plotting
# plt.figure(figsize=(10, 6))

# # Calculate and plot the diffraction limit for each diameter
# for diameter in D:
#     sizes = diffraction_limit(diameter, distances)
#     plt.plot(distances / 1e3, sizes, label=f'Diameter: {diameter*100:.0f}cm')  # Convert distances to km

# # Add horizontal lines for object sizes with labels
# dim_obj = [0.3, 3, 5, 10, 25]
# labels = ['6U Cubesat', 'Iridium', 'Starlink', 'Sentinel', 'Envisat']

# #for obj_size, label in zip(dim_obj, labels):
# #    plt.axhline(obj_size, color='gray', linestyle='--', alpha=0.7)
# #    plt.text(distances[-1] / 1e3, obj_size, f'  {label}', verticalalignment='center', horizontalalignment='right')

# #for obj_size, label in zip(dim_obj, labels):
# #    plt.axhline(obj_size, color='black', linestyle='--', alpha=0.7)
# #    plt.text(distances[-1] / 1e3, obj_size + obj_size * 0.05, f'  {label}', verticalalignment='center', horizontalalignment='right')

# x_offset = distances[-1] * 0.0000000001  # Adding 1% of the total x-range as an offset
# for obj_size, label in zip(dim_obj, labels):
#     plt.axhline(obj_size, color='black', linestyle='--', alpha=0.7)
#     plt.text(distances[-1] / 1e3 + x_offset, obj_size +  0.5, f'{label}', verticalalignment='center', horizontalalignment='left')
#     plt.xlim(0,1100)


# plt.xlabel('Distance (km)')
# plt.ylabel('Detectable Size (m)')
# plt.title('Detectable Size vs Distance')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# for diameter in D:
#     howfar = dist_detect(diameter, dim_obj)
#     print(f"Diameter: {diameter*100:.0f}cm -> Resolvable Distances: {howfar/1000}")




# #TEST HOW FAR

# payload=OpticPayload()
# DIAMETER=0.09 #Diameter of the optic to fit in a Cubesat'
# WAVELENGTH = 700e-9 #Upper limit of visible light
# theta= 1.22 * WAVELENGTH / DIAMETER 
# size=10
# #distance=1000*1000 #It has to be in meters
# distance_s= size/(2*np.tan(np.rad2deg(theta)/2))
# distance_s= size/(2*np.tan((theta)/2))

# print (f"the distance is {distance_s}")
# howfar = payload.dist_detect()

#canI=payload.possibility_observation(distance)
#print(howfar)
#print(canI)
#print(f"Diameter: {Diameter*100:.0f}cm -> Resolvable Distances: {howfar/1000}")