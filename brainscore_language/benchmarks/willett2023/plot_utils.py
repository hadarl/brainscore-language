import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from matplotlib import colors


electrode_mapping = \
               [192, 193, 208, 216, 160, 165, 178, 185,      62,  51,  43,  35,  94,  87,  79,  78,
                194, 195, 209, 217, 162, 167, 180, 184,      60,  53,  41,  33,  95,  86,  77,  76,
         		196, 197, 211, 218, 164, 170, 177, 189,      63,  54,  47,  44,  93,  84,  75,  74,
				198, 199, 210, 219, 166, 174, 173, 187,      58,  55,  48,  40,  92,  85,  73,  72,
				200, 201, 213, 220, 168, 176, 183, 186,      59,  45,  46,  38,  91,  82,  71,  70,
				202, 203, 212, 221, 172, 175, 182, 191,      61,  49,  42,  36,  90,  83,  69,  68,
				204, 205, 214, 223, 161, 169, 181, 188,      56,  52,  39,  34,  89,  81,  67,  66,
				206, 207, 215, 222, 163, 171, 179, 190,      57,  50,  37,  32,  88,  80,  65,  64,
                129, 144, 150, 158, 224, 232, 239, 255,     125, 126, 112, 103,  31,  28,  11,   8,
				128, 142, 152, 145, 226, 233, 242, 241,     123, 124, 110, 102,  29,  26,   9,   5,
				130, 135, 148, 149, 225, 234, 244, 243,     121, 122, 109, 101,  27,  19,  18,   4,
				131, 138, 141, 151, 227, 235, 246, 245,     119, 120, 108, 100,  25,  15,  12,   6,
				134, 140, 143, 153, 228, 236, 248, 247,     117, 118, 107,  99,  23,  13,  10,   3,
				132, 146, 147, 155, 229, 237, 250, 249,     115, 116, 106,  97,  21,  20,   7,   2,
				133, 137, 154, 157, 230, 238, 252, 251,     113, 114, 105,  98,  17,  24,  14,   0,
				136, 139, 156, 159, 231, 240, 254, 253,     127, 111, 104,  96,  30,  22,  16,   1]


def plot_score_vs_layer(layer_scores_ordered, layer_scores_std_ordered):

    import matplotlib.pyplot as plt

    layer_names = []
    for layer in tqdm([f'transformer.h.{block}.ln_1' for block in range(6)], desc='layers'):
        layer_names.append(layer)

    fig, ax = plt.subplots(figsize=(10,10))
    x = np.arange(len(layer_scores_ordered))
    #ax.scatter(x, layer_scores_ordered)
    ax.errorbar(x, layer_scores_ordered, yerr=layer_scores_std_ordered, fmt='o')
    ax.set_xticks(x)
    fig.subplots_adjust(bottom=0.3, left=0.2)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_ylabel('score')
    # ax.legend()
    #plt.show()

    #fig.savefig('Willett2023_gpt2-large.png')

def plot_activations_per_word(assembly):

    words = assembly.word.values

    # Create subplots for each word
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 8))

    # Flatten the axes for easier indexing
    axes = axes.flatten()

    for i, word in enumerate(words):
        ax = axes[i]
        data_for_word = assembly.sel(word=word)

        # Plot the data as a colored map
        im = ax.imshow(data_for_word.values.T, cmap='viridis', aspect='auto', origin='lower')

        # Add labels and title
        ax.set_title(f"Word: {word}")
        ax.set_xlabel('Neuroid Index')
        ax.set_ylabel('Time Bin')

    # Add a colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', pad=0.02)
    cbar.set_label('Mean Activity')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def plot_histogram_neuroid(assembly, index):

    # Extract values for the 3rd neuroid
    neuroid_values = assembly.sel(neuroid=index).values.flatten()

    # Create a histogram
    plt.hist(neuroid_values, bins=50, color='skyblue', edgecolor='black')

    # Add labels and title
    plt.xlabel('Neuroid Values')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Values for Neuroid {index}')

    # Show the plot
    plt.show()

def plot_scores_per_array(layer_scores_per_neuroid, electrode_mapping):

    Nr = 2
    Nc = 2

    fig, axs = plt.subplots(Nr, Nc)
    fig.suptitle('Scores per array')


    images = []
    for i in range(Nr):
        for j in range(Nc):
            # Generate data with a range that varies from one plot to the next.
            array_ind = i*2 + j
            #data = electrode_mapping
            data = layer_scores_per_neuroid[array_ind][(array_ind*64):(array_ind+1)*64].reshape(8,8)
            images.append(axs[i, j].imshow(data))
            axs[i, j].label_outer()

    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)


    # Make images respond to changes in the norm of other images (e.g. via the
    # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
    # recurse infinitely!
    def update(changed_image):
        for im in images:
            if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())


    for im in images:
        im.callbacks.connect('changed', update)

    plt.show()


def plot_scores_per_layer(layer_scores_per_neuroid_reordered, filename):

    #im = Image.new('RGBA', (400, 400), (0, 255, 0, 255))
    # draw = ImageDraw.Draw(im)
    #draw.line((100, 200, 150, 300), fill=128)
    #im.show()

    Nr = 2
    Nc = 3

    fig, axs = plt.subplots(Nr, Nc)
    fig.suptitle('Scores per layer')


    images = []
    for i in range(Nr):
        for j in range(Nc):
            # Generate data with a range that varies from one plot to the next.
            layer_ind = i*Nc + j
            #data = electrode_mapping
            data = layer_scores_per_neuroid_reordered[layer_ind].reshape(16,16)
            images.append(axs[i, j].imshow(data))
            axs[i, j].label_outer()


    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)


    # Make images respond to changes in the norm of other images (e.g. via the
    # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
    # recurse infinitely!
    def update(changed_image):
        for im in images:
            if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())


    for im in images:
        im.callbacks.connect('changed', update)

    plt.show()

    fig.savefig(filename)
    # 'Willett2023_gpt2 - large_mapPerLayer_allFeatures.png'