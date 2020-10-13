import cv2
from scipy import ndimage
from scipy.signal import convolve2d
from skimage.morphology import remove_small_objects, skeletonize
from skimage.measure import label, regionprops
from skimage.filters import threshold_local, threshold_otsu
from deprecated_demos.ta.wshed import Wshed  # TODO just copy the relevant part of it
from epyseg.binarytools.cell_center_detector import add_2d_border
from epyseg.img import Img
import matplotlib.pyplot as plt
import numpy as np
from epyseg.postprocess.superpixel_methods import split_into_vertices_and_bonds, \
    detect_unconnected_bonds  # TODO will need push epyseg first... --> is that ok --> maybe anyway I need it
from epyseg.tools.logger import TA_logger
logger = TA_logger()

__DEBUG = False
__VISUAL_DEBUG = False

def detect_vertices(img):
    patterns = []
    pattern1 = np.asarray([[1, 0, 0],
                           [0, 1, 1],
                           [1, 0, 0]])
    patterns.append(pattern1)
    pattern2 = np.rot90(pattern1)
    patterns.append(pattern2)
    pattern3 = np.rot90(pattern2)
    patterns.append(pattern3)
    pattern4 = np.rot90(pattern3)
    patterns.append(pattern4)

    pattern7 = np.asarray([[0, 0, 1],
                           [1, 1, 0],
                           [0, 1, 0]])
    patterns.append(pattern7)
    pattern8 = np.rot90(pattern7)
    patterns.append(pattern8)
    pattern9 = np.rot90(pattern8)
    patterns.append(pattern9)
    pattern10 = np.rot90(pattern9)
    patterns.append(pattern10)

    pattern11 = np.asarray([[0, 0, 0],
                            [1, 1, 1],
                            [0, 1, 0]])
    patterns.append(pattern11)
    pattern12 = np.rot90(pattern11)
    patterns.append(pattern12)
    pattern14 = np.rot90(pattern12)
    patterns.append(pattern14)
    pattern15 = np.rot90(pattern14)
    patterns.append(pattern15)

    pattern5 = np.asarray([[1, 0, 1],
                           [0, 1, 0],
                           [1, 0, 1]])
    patterns.append(pattern5)

    pattern6 = np.asarray([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]])
    patterns.append(pattern6)

    SKEL_PATTERNS = True
    if SKEL_PATTERNS:
        pattern16 = np.asarray([[0, 0, 1],
                                [1, 1, 1],
                                [0, 1, 0]])
        patterns.append(pattern16)
        pattern17 = np.rot90(pattern16)
        patterns.append(pattern17)
        pattern18 = np.rot90(pattern17)
        patterns.append(pattern18)
        pattern19 = np.rot90(pattern18)
        patterns.append(pattern19)

        pattern20 = np.asarray([[1, 0, 0],
                                [1, 1, 1],
                                [0, 1, 0]])
        patterns.append(pattern20)
        pattern21 = np.rot90(pattern20)
        patterns.append(pattern21)
        pattern22 = np.rot90(pattern21)
        patterns.append(pattern22)
        pattern23 = np.rot90(pattern22)
        patterns.append(pattern23)

        pattern24 = np.asarray([[0, 1, 0],
                                [0, 1, 1],
                                [1, 0, 1]])
        patterns.append(pattern24)
        pattern25 = np.rot90(pattern24)
        patterns.append(pattern25)
        pattern26 = np.rot90(pattern25)
        patterns.append(pattern26)
        pattern27 = np.rot90(pattern26)
        patterns.append(pattern27)

        pattern28 = np.asarray([[0, 1, 0],
                                [1, 1, 0],
                                [1, 0, 1]])
        patterns.append(pattern28)
        pattern29 = np.rot90(pattern28)
        patterns.append(pattern29)
        pattern30 = np.rot90(pattern29)
        patterns.append(pattern30)
        pattern31 = np.rot90(pattern30)
        patterns.append(pattern31)

        pattern32 = np.asarray([[1, 0, 0],
                                [0, 1, 0],
                                [1, 0, 1]])
        patterns.append(pattern32)
        pattern33 = np.rot90(pattern32)
        patterns.append(pattern33)
        pattern34 = np.rot90(pattern33)
        patterns.append(pattern34)
        pattern35 = np.rot90(pattern34)
        patterns.append(pattern35)

    vertices = np.zeros_like(img, dtype=np.uint8)
    type = img.dtype

    # detect square blocks artifact of skeletonize and replace by 4 vertices
    pattern_2_per_2_square_block = np.asarray([[1, 1],
                                               [1, 1]])
    for j in range(1, img.shape[0] - 1, 1):
        for i in range(1, img.shape[1] - 1, 1):
            if img[j][i] == 1:
                array = img[j - 1:j + 1, i - 1:i + 1]
                if np.all(array == pattern_2_per_2_square_block.astype(type)):
                    vertices[j - 1:j + 1, i - 1:i + 1] = 255

    for j in range(1, img.shape[0] - 1, 1):
        for i in range(1, img.shape[1] - 1, 1):
            if img[j][i] == 1 and vertices[j][i] != 255:
                array = img[j - 1:j + 2, i - 1:i + 2]
                for pattern in patterns:
                    if np.all(array == pattern.astype(type)):
                        vertices[j][i] = 255
    return vertices


def find_neurons(global_threshold, neuron_minimum_size_threshold=45):
    global_threshold = add_2d_border(global_threshold)

    global_threshold = ~remove_small_objects(~global_threshold.astype(np.bool), min_size=10, connectivity=1,
                                             in_place=False)
    skel_or_shed = skeletonize(global_threshold)

    real_neurons_above_size = remove_small_objects(skel_or_shed, min_size=neuron_minimum_size_threshold, connectivity=2,
                                                   in_place=False)
    real_neurons_above_size = ndimage.binary_dilation(real_neurons_above_size)

    if __VISUAL_DEBUG:
        plt.imshow(real_neurons_above_size.astype(np.uint8) + skel_or_shed.astype(
            np.uint8))
        plt.title('real')
        plt.show()

    rgb_image = np.zeros(shape=(*real_neurons_above_size.shape, 3), dtype=np.uint8)
    rgb_image[..., 0] = skel_or_shed.astype(np.uint8) * 255
    rgb_image[..., 1] = real_neurons_above_size.astype(np.uint8) * 255
    rgb_image[..., 2] = skel_or_shed.astype(np.uint8) * 255
    if __VISUAL_DEBUG:
        plt.imshow(rgb_image)
        plt.show()


def detect_cell_bonds(global_threshold):
    global_threshold = add_2d_border(global_threshold)

    global_threshold = ~remove_small_objects(~global_threshold.astype(np.bool), min_size=10, connectivity=1,
                                             in_place=False)
    skel_or_shed = skeletonize(global_threshold)

    vertices_quick, cut_bonds_quick = split_into_vertices_and_bonds(
        skel_or_shed)

    if __VISUAL_DEBUG:
        plt.imshow(vertices_quick)
        plt.show()

        plt.imshow(cut_bonds_quick)
        plt.show()

    kernel = np.ones((3, 3))
    mask = convolve2d(skel_or_shed, kernel, mode='same', fillvalue=1)

    real_vx = mask.copy()

    mask[mask < 4] = 0
    mask[mask >= 4] = 1

    mask = np.logical_and(mask, skel_or_shed).astype(np.uint8)
    real_vx[mask == 0] = 0

    if __VISUAL_DEBUG:
        plt.imshow(real_vx)
        plt.title('real_vx')
        plt.show()

    vertices = np.zeros_like(real_vx)
    vertices[real_vx < 5] = 0
    vertices[real_vx >= 5] = 1
    if __VISUAL_DEBUG:
        plt.imshow(vertices)
        plt.title('vertices')
        plt.show()

    if __DEBUG:
        Img(vertices.astype(np.uint8) * 255, dimensions='hw').save(
            '/home/aigouy/Bureau/trash/trash4/vertices.tif')
        Img((vertices_quick - vertices).astype(np.uint8) * 255, dimensions='hw').save(
            '/home/aigouy/Bureau/trash/trash4/small_pieces_of_bonds.tif')
        Img(skel_or_shed.astype(np.uint8) * 255, dimensions='hw').save(
            '/home/aigouy/Bureau/trash/trash4/skel.tif')
        Img(cut_bonds_quick.astype(np.uint8) * 255, dimensions='hw').save(
            '/home/aigouy/Bureau/trash/trash4/cut_bonds_quick.tif')
        Img(real_vx.astype(np.uint8), dimensions='hw').save(
            '/home/aigouy/Bureau/trash/trash4/real_vx.tif')

    labels_quick = label(cut_bonds_quick, connectivity=2, background=0)
    if __DEBUG:
        Img(labels_quick.astype(np.uint8), dimensions='hw').save(
            '/home/aigouy/Bureau/trash/trash4/labels_quick.tif')

    vertices2 = detect_vertices(skel_or_shed)
    if __DEBUG:
        Img(vertices2.astype(np.uint8), dimensions='hw').save(
            '/home/aigouy/Bureau/trash/trash4/vertices2.tif')

    vertices_quick[vertices2 != 0] = 0

    if __VISUAL_DEBUG:
        plt.imshow(vertices_quick)
        plt.title('test to reconnect')
        plt.show()

    if __DEBUG:
        Img(vertices_quick.astype(np.uint8) * 255, dimensions='hw').save(
            '/home/aigouy/Bureau/trash/trash4/bonds_to_reconnect.tif')

    for region in regionprops(labels_quick):
        for coordinates in region.coords:
            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    try:
                        if vertices_quick[coordinates[0] + i, coordinates[1] + j] != 0:
                            labels_quick[coordinates[0] + i, coordinates[1] + j] = region.label
                            vertices_quick[coordinates[0] + i, coordinates[1] + j] = 0
                    except:
                        pass

    if __DEBUG:
        Img(labels_quick.astype(np.uint8), dimensions='hw').save(
            '/home/aigouy/Bureau/trash/trash4/bonds_reconnected.tif')

        Img(vertices_quick.astype(np.uint8), dimensions='hw').save(
            '/home/aigouy/Bureau/trash/trash4/unconnected_remaining.tif')

    labels_vertices_quick = label(vertices_quick, connectivity=1, background=0)
    for region in regionprops(labels_vertices_quick):
        for coordinates in region.coords:
            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    try:
                        if labels_quick[coordinates[0] + i, coordinates[1] + j] != 0:
                            labels_quick[coordinates[0], coordinates[1]] = labels_quick.max() + 1
                    except:
                        pass

    labels_quick[vertices2 != 0] = 255

    if __DEBUG:
        Img(labels_quick.astype(np.uint8), dimensions='hw').save(
            '/home/aigouy/Bureau/trash/trash4/bonds_reconnected_with_vertices.tif')


def prune_dendrites(skel_or_shed, labels_quick, prune_below=3):
    if prune_below > 0:
        unconnected = detect_unconnected_bonds(skel_or_shed)

        if __DEBUG:
            Img(unconnected, dimensions='hw').save('/home/aigouy/Bureau/trash/trash4/unconnected.tif')
        bonds_allowed_to_be_pruned = []

        labels_unconnected = label(unconnected, connectivity=1, background=0)
        for region in regionprops(labels_unconnected):
            for coordinates in region.coords:
                if labels_quick[coordinates[0], coordinates[1]] != 0:
                    bonds_allowed_to_be_pruned.append(labels_quick[coordinates[0], coordinates[1]])

        for region in regionprops(labels_quick):
            if region.label not in bonds_allowed_to_be_pruned:
                continue
            if region.area <= prune_below:
                for coordinates in region.coords:
                    labels_quick[coordinates[0], coordinates[1]] = 0

        if __DEBUG:
            Img(labels_quick.astype(np.uint8), dimensions='hw').save(
                '/home/aigouy/Bureau/trash/trash4/pruned.tif')

def threshold_neuron(neuron, mode='global',blur_method='mean', spin_value=6):
    if neuron is None:
        logger.error('Please load an image first')
        return None

    global_threshold = None

    if __VISUAL_DEBUG:
        plt.imshow(neuron)
        plt.show()

    if 'global' in mode:
        if 'ean' in blur_method.lower():
            mean = neuron.mean() + spin_value
            global_threshold = neuron.copy()
            global_threshold[global_threshold < mean] = 0
            global_threshold[global_threshold >= mean] = 255

            if __VISUAL_DEBUG:
                plt.imshow(global_threshold)
                plt.show()
        else:
            median = np.median(neuron)
            global_threshold = neuron.copy()
            global_threshold[global_threshold < median + spin_value] = 0
            global_threshold[global_threshold >= median + spin_value] = 255

            if __VISUAL_DEBUG:
                plt.imshow(global_threshold)
                plt.title('median global')
                plt.show()

    local = None
    if 'local' in mode:
        if 'ean' in blur_method.lower():
            func = lambda arr: arr.mean() + spin_value
            local = neuron > threshold_local(neuron, 31, 'generic', param=func)
        else:
            func = lambda arr: np.median(arr) + spin_value
            local = neuron > threshold_local(neuron, 31, 'generic', param=func)

    if local is not None and global_threshold is not None:
        local_and_global = np.logical_and(global_threshold, local)
        if __VISUAL_DEBUG:
            plt.imshow(local_and_global)
            plt.title('local and global')
            plt.show()
        return local_and_global

    if global_threshold is not None:
        return remove_small_objects(global_threshold, min_size=10, connectivity=2, in_place=False)
    else:
        return remove_small_objects(local, min_size=10, connectivity=2, in_place=False)




def watershed_segment_neuron(neuron, local_and_global, fillSize=10, autoSkel=True, first_blur=2.1, second_blur=1.4,
                             min_size=10):
    test = Wshed.run(neuron, first_blur=first_blur, second_blur=second_blur, min_size=min_size)

    if __VISUAL_DEBUG:
        plt.imshow(test)
        plt.title('shed on neurons')
        plt.show()

    # §§§§§§!!!!!!!!!!!!!!!!! nb smaller than minsize so set it to 2 minimum or do +1 TO IT

    # fixed size

    # TODO only keep wshed for mask --> important
    test2 = np.logical_and(test, local_and_global)
    test2 = remove_small_objects(test2, min_size=2, connectivity=2, in_place=False)
    test2 = ~remove_small_objects(~test2, min_size=10, connectivity=1, in_place=False)
    test2 = skeletonize(test2)
    test2 = remove_small_objects(test2, min_size=20, connectivity=2, in_place=False)
    if __VISUAL_DEBUG:
        plt.imshow(test2)
        plt.title('shed on neurons with mask')
        plt.show()

    if fillSize > 0:
        filteredWshed = ~remove_small_objects(~test2, min_size=fillSize, connectivity=1, in_place=False)
    else:
        filteredWshed = test2
    if autoSkel:
        filteredWshed = skeletonize(filteredWshed)

    if __VISUAL_DEBUG:
        plt.imshow(filteredWshed)
        plt.title('mega final')
        plt.show()


def detect_cell_body(neuron, fillHoles=300, denoise=600, nbOfErosions=2, nbOfDilatations=2, extraCutOff=5):
    threshold_value = threshold_otsu(neuron)
    cell_body = neuron.copy()
    cell_body[cell_body < threshold_value+extraCutOff] = 0
    cell_body[cell_body >= threshold_value+extraCutOff] = 1

    cell_body = remove_small_objects(cell_body.astype(np.bool), min_size=denoise, connectivity=2, in_place=False)

    if fillHoles > 0:
        cell_body = ~remove_small_objects(~cell_body, min_size=fillHoles, connectivity=1, in_place=False)

    s = ndimage.generate_binary_structure(2, 1)

    for i in range(0, nbOfErosions+1, 1):
        cell_body = ndimage.binary_erosion(cell_body, structure=s)

    for i in range(0, nbOfDilatations+1, 1):
        cell_body = ndimage.binary_dilation(cell_body, structure=s)

    cell_body = remove_small_objects(cell_body, min_size=denoise, connectivity=2, in_place=False)

    if __VISUAL_DEBUG:
        plt.imshow(cell_body)
        plt.title('cell body')
        plt.show()

    contours, hierarchy = cv2.findContours(cell_body.astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # draw all the countours
    backtorgb = cv2.cvtColor(np.zeros_like(cell_body).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    cv2.drawContours(backtorgb, contours, -1, (0, 255, 0), 1)

    backtorgb = backtorgb[..., 1]

    if __VISUAL_DEBUG:
        plt.imshow(backtorgb)
        plt.show()

    final_seeds = label(cell_body, connectivity=1, background=0)
    if __VISUAL_DEBUG:
        plt.imshow(final_seeds)
        plt.title('test')
        plt.show()

    for region in regionprops(final_seeds):
        for coordinates in region.coords:
            neuron[coordinates[0], coordinates[1]] = 0
            if backtorgb[coordinates[0], coordinates[1]] != 0:
                neuron[coordinates[0], coordinates[1]] = 255

    if __VISUAL_DEBUG:
        plt.imshow(neuron)
        plt.show()


if __name__ == '__main__':
    neuron = Img('/home/aigouy/Dropbox/amrutha/Opto60x_7x1.5µm_30s_250ms_72-z596.png')
    # use this to test the functions and maybe compare to other version

