from cProfile import label
from tabnanny import check
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy import stats
####################################
### FIXED KEY TEMPLATE BASED DPA ###
####################################

# Load in data
all_traces = np.load(r'D:\School\Master\Vakken\Jaar 2\Thesis spul\wetransfer_chipwhisperer_2022-01-20_1217\Chipwhisperer\traces.npy')[:10000]
all_ptext  = np.load(r'D:\School\Master\Vakken\Jaar 2\Thesis spul\wetransfer_chipwhisperer_2022-01-20_1217\Chipwhisperer\plain.npy')
all_label = np.load(r'D:\School\Master\Vakken\Jaar 2\Thesis spul\wetransfer_chipwhisperer_2022-01-20_1217\Chipwhisperer\labels.npy')


temp_traces = all_traces[0:8000]
temp_ptext = all_ptext[0:8000]
temp_label = all_label[0:8000]


# Plot and print to check data

plt.plot(temp_traces[0])
print(temp_ptext[0])
#plt.show(block=True)


# Simple S-Box lookup table
sbox=(
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16) 

# Calculate intermediate value by lookup input XOR key for a certain byte index
def intermediate_val(input, key, byte_idx=0):
    return sbox[input[byte_idx] ^ key]

HW = [bin(a).count("1") for a in range(256)]

# Calculate Hamming weight of sbox output
def get_hamming_weight(input, key, byte_idx=0):
    return HW[intermediate_val(input, key, byte_idx)]

# Create HW groups to group traces based on  HW 
hw_trace_groups = [[] for _ in range(9)]

for i in range (len(temp_traces)):
    hw = HW[temp_label[i]] 
    hw_trace_groups[hw].append(temp_traces[i])

# POI selection, reducing the dimensionality => increases efficiency and speed
means = [[] for _ in range(9)]
var = [[] for _ in range(9)]
counts = [[] for _ in range(9)]
coefficients = np.zeros(len(temp_traces[0]))

# Calculate means and variance of each label
for i in range(len(hw_trace_groups)):
    means[i] = np.mean(hw_trace_groups[i], axis=0)
    var[i] = np.var(hw_trace_groups[i], axis=0)
    counts[i] = len(hw_trace_groups[i])

# Calculate coefficients, with sum of squared differences
for i in range(9):
    for j in range(i):
        temp = pow(means[i] - means[j], 2)
        temp /= (var[i]/counts[i] + var[j]/counts[j])
        coefficients += temp

# select top-n coefficients to represent the POI's
n_poi = 2
sorted =np.argsort(-coefficients)
relevant_indices = np.zeros(n_poi, dtype=np.int16)

poi_spacing = 20 
dist = 0

for i in range(n_poi):
    if i > 0:
        check_idx = 1

        while (dist <= poi_spacing):
            last_poi = relevant_indices[i-1]
            cur_poi = sorted[check_idx]
            dist = np.abs(cur_poi - last_poi)
            check_idx+=1

        relevant_indices[i] = cur_poi
        
    else:
        relevant_indices[0] = sorted[0]
    
    
        


print(relevant_indices)
# Plot some traces to see their POI's
fig, axs = plt.subplots(4, sharey=True)
axs[0].plot(temp_traces[0], '-bo', markevery=relevant_indices, markersize=6, markerfacecolor='k')
axs[1].plot(temp_traces[1], '-go', markevery=relevant_indices, markersize=6, markerfacecolor='k')
axs[2].plot(temp_traces[2], '-ro', markevery=relevant_indices, markersize=6, markerfacecolor='k')
axs[3].plot(temp_traces[3], '-co', markevery=relevant_indices, markersize=6, markerfacecolor='k')

plt.show(block=True)

# Create templates of the selected POI's
template_means = [np.array([]) for _ in range(9)]

# Create mean vectors and covariance matrices for each POI for each HW
for hw in range(len(hw_trace_groups)):
    poi_traces = np.array(hw_trace_groups[hw])[:, relevant_indices]
    template_means[hw] = np.mean(poi_traces, axis=0)

template_cov_matrix = np.zeros((9, n_poi, n_poi))

for hw in range(9):
    for i in range(n_poi):
        for j in range(n_poi):            
            x = np.array(hw_trace_groups[hw])[:, relevant_indices[i]]
            y = np.array(hw_trace_groups[hw])[:, relevant_indices[j]]
            cov = np.cov(x,y)
            template_cov_matrix[hw, i, j] = cov[0][1]


##################
# Perform attack #
##################

# Load in attack stuff

atk_traces = all_traces[8001:10000]
atk_ptext = all_ptext[8001:108000]

actual_key = np.load(r'courses\sca101\traces\lab4_2_key.npy')
print(actual_key)

# Create 
guess_proba = np.zeros(256)
attack_byte = 1

for i in range(len(atk_traces)):
    cur_trace = atk_traces[i, relevant_indices]

    for key in range(256):
        hw = get_hamming_weight(atk_ptext[i], key, attack_byte)
        rv = multivariate_normal(template_means[hw], template_cov_matrix[hw], allow_singular=True)

        proba_key_guess = rv.pdf(cur_trace)
        guess_proba[key] += np.log(proba_key_guess)

    print(np.argsort(guess_proba)[-5:])
'''
P_k = np.zeros(256)
for j in range(len(atk_traces)):
    # Grab key points and put them in a matrix
    a = [atk_traces[j][relevant_indices[i]] for i in range(len(relevant_indices))]
    
    # Test each key
    for k in range(256):
        # Find HW coming out of sbox
        hw = get_hamming_weight(atk_ptext[j], k, attack_byte)
    
        # Find p_{k,j}
        rv = multivariate_normal(template_means[hw], template_cov_matrix[hw], allow_singular=False)
        p_kj = rv.pdf(a)
   
        # Add it to running total
        P_k[k] += np.log(p_kj)

    # Print our top 5 results so far
    # Best match on the right
    print(P_k.argsort()[-5:])
    '''