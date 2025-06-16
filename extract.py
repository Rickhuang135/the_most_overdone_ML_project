import struct
import asyncio
import png

def extract_images(file_path, number =None):
    with open(file_path, 'rb') as file:
        magic_number = struct.unpack('>I', file.read(4))[0]
        num_images = struct.unpack('>I', file.read(4))[0]
        num_rows = struct.unpack('>I', file.read(4))[0]
        num_cols = struct.unpack('>I', file.read(4))[0]
        if number is None:
            number = num_images
        images = []
        for _ in range(number):
            image = []
            for _ in range(num_rows):
                row = []
                for _ in range(num_cols):
                    pixel = struct.unpack('>B', file.read(1))[0]
                    row.append(pixel)
                image.append(row)
            images.append(image)
        
        return images
    

def to_png(read_path, write_path, number=None):
    with open(read_path, 'rb') as file:
        magic_number = struct.unpack('>I', file.read(4))[0]
        num_images = struct.unpack('>I', file.read(4))[0]
        num_rows = struct.unpack('>I', file.read(4))[0]
        num_cols = struct.unpack('>I', file.read(4))[0]
        if number is None:
            number = num_images
        for i in range(number):
            image = []
            for _ in range(num_rows):
                row = []
                for _ in range(num_cols):
                    pixel = struct.unpack('>B', file.read(1))[0]
                    row.append(pixel)
                image.append(row)
            asyncio.run(write_png(f"{write_path}/image{i}.png", image))
            
async def write_png(file_name,image:list):
    writer = png.Writer(width = len(image), height = len(image[0]), bitdepth=8, greyscale=True)
    with open(file_name, "wb") as file:
        writer.write(file, image)