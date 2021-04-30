import PIL.Image
import numpy

def wrong_pixels(ground_truth, inference_mask):
    return (ground_truth-1) != inference_mask


def map_wrong_pixels(img, ground_truth, pred):

    # Fill wrong pixel with red rgb and 0.5 alpha
    #mask = numpy.full((img.shape[0], img.shape[1], 3), (255, 0, 51)).astype(numpy.uint8)
    #mask = numpy.dstack( ( mask, (numpy.where(wrong_pixels(ground_truth, pred), 0.5, 0)*255).astype(numpy.uint8) ) )
    #mask = numpy.where(wrong_pixels(ground_truth, pred), (255, 0, 51, 0.5*255), (0, 0, 0, 0)).astype(numpy.uint8)
    mask = numpy.zeros((img.shape[0], img.shape[1], 4))
    mask[ wrong_pixels(ground_truth, pred) ] = (255, 0, 51, 0.5*255)

    # agregates images
    Image_img = PIL.Image.fromarray(img.astype(numpy.uint8)).convert('RGBA')
    Image_mask = PIL.Image.fromarray(mask.astype(numpy.uint8), mode='RGBA')

    res = PIL.Image.alpha_composite(Image_img, Image_mask)

    return res

def visualize_wrong_pixels(img, ground_truth, pred, save='', show=True):

    masked_image = map_wrong_pixels(img, ground_truth, pred)

    if( save != '' ):
        masked_image.save(save)

    if( show ):
        masked_image.show()

def map_mask(img, pred, pallette):

    mask = numpy.zeros((pred.shape[0], pred.shape[1], 4))
    for index in numpy.unique(pred):
        channels = list(pallette[index])
        channels.append(0.5*255)
        mask[ pred == index ] = channels

    mask = mask.astype(numpy.uint8)

    # agregates images
    Image_img = PIL.Image.fromarray(img.astype(numpy.uint8)).convert('RGBA')
    Image_mask = PIL.Image.fromarray(mask, mode='RGBA')

    masked_image = PIL.Image.alpha_composite(Image_img, Image_mask)

    return masked_image

def visualize_mask(img, pred, pallette, save='', show=True):

    masked_image = map_mask(img, pred, pallette)

    if( save != '' ):
        masked_image.save(save)

    if( show ):
        masked_image.show()
