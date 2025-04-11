from PIL import Image, ImageOps, ImageFilter, ImageEnhance

def resize(image: Image.Image, width: int, height: int) -> Image.Image:
    """
    Resize the image to the specified width and height.

    Args:
        image (Image.Image): The input image.
        width (int): The target width.
        height (int): The target height.

    Returns:
        Image.Image: The resized image.
    """
    return image.resize((width, height), Image.LANCZOS)

def change_color_depth(image: Image.Image, color_depth: int) -> Image.Image:
    """
    Change the color depth of the image by posterizing it.

    This function reduces the number of bits per channel.
    Note: color_depth should be between 1 and 8.

    Args:
        image (Image.Image): The input image.
        color_depth (int): The number of bits to keep per channel.

    Returns:
        Image.Image: The image with the modified color depth.
    """
    if image.mode != 'RGB':
        image = image.convert("RGB")
    return ImageOps.posterize(image, bits=color_depth)

def crop(image: Image.Image, crop_width: int, crop_height: int) -> Image.Image:
    """
    Crop the image to the specified width and height using center cropping.

    Args:
        image (Image.Image): The input image.
        crop_width (int): The width of the crop region.
        crop_height (int): The height of the crop region.

    Returns:
        Image.Image: The center-cropped image.

    Raises:
        ValueError: If the crop dimensions exceed the image dimensions.
    """
    if crop_width > image.width or crop_height > image.height:
        raise ValueError("Crop dimensions exceed image dimensions")
    
    left = (image.width - crop_width) // 2
    top = (image.height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    return image.crop((left, top, right, bottom))

def rotate(image: Image.Image, angle: float, expand: bool = True) -> Image.Image:
    """
    Rotate the image by the specified angle.

    Args:
        image (Image.Image): The input image.
        angle (float): The angle (in degrees) to rotate the image. Positive values
                       rotate counter-clockwise.
        expand (bool): Whether to expand the output image to ensure the rotated image is contained fully.

    Returns:
        Image.Image: The rotated image.
    """
    return image.rotate(angle, expand=expand)

def flip_horizontal(image: Image.Image) -> Image.Image:
    """
    Flip the image horizontally.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The horizontally flipped image.
    """
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def flip_vertical(image: Image.Image) -> Image.Image:
    """
    Flip the image vertically.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The vertically flipped image.
    """
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def convert_to_grayscale(image: Image.Image) -> Image.Image:
    """
    Convert the image to grayscale.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The grayscale version of the image.
    """
    return image.convert("L")

def apply_sharpen(image: Image.Image) -> Image.Image:
    """
    Apply a sharpen filter to the image.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The sharpened image.
    """
    return image.filter(ImageFilter.SHARPEN)

def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    """
    Adjust the brightness of the image.

    Args:
        image (Image.Image): The input image.
        factor (float): A factor where 1.0 gives the original image,
                        less than 1.0 darkens the image, and greater than 1.0 brightens it.

    Returns:
        Image.Image: The brightness-adjusted image.
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    """
    Adjust the contrast of the image.

    Args:
        image (Image.Image): The input image.
        factor (float): A factor where 1.0 gives the original image,
                        less than 1.0 decreases contrast, and greater than 1.0 increases it.

    Returns:
        Image.Image: The contrast-adjusted image.
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def apply_blur(image: Image.Image, radius: float = 2.0) -> Image.Image:
    """
    Apply a Gaussian blur to the image.

    Args:
        image (Image.Image): The input image.
        radius (float): The blur radius. Higher values produce a more blurred image.

    Returns:
        Image.Image: The blurred image.
    """
    return image.filter(ImageFilter.GaussianBlur(radius))
