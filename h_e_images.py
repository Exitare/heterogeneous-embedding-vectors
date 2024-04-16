import slideio
from pathlib import Path
import matplotlib.pyplot as plt

# load the slide


# iterate thorugh data/images folder
for file in Path("data", "images").iterdir():
    print(file.suffix)
    if not file.suffix.endswith(".svs"):
        continue
    slide = slideio.open_slide(
        str(Path(file)),"SVS")
    print(slide.num_aux_images)
    # get image data
    scene = slide.get_scene(0)
    print(scene.name, scene.rect, scene.num_channels, scene.resolution)
    print(scene.rect)
    image = scene.read_block(size=(500,0))
    #plt.imshow(image)
    #plt.show()
