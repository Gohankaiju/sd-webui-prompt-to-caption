# sd-webui-prompt-to-caption

## Introduction
 - Extract prompt data from images in the specified folder and create a caption text file for study.

 - If metadata could not be obtained from the images, a caption file can be created by BLIP instead.

## How to Use
### Prompt
![2](https://github.com/Gohankaiju/sd-webui-prompt-to-caption/assets/167270541/99ed0d60-e54e-4d02-972b-d5ec41ec284e)

 - Specify the path of image folder and click the gen button.

 - If "Enable BLIP" is checked, BLIP caption will be created on images for which metadata could not be obtained. (If Enable BLIP is not checked, an empty text file will be created.)

### BLIP
![1](https://github.com/Gohankaiju/sd-webui-prompt-to-caption/assets/167270541/9a98f94c-d465-477a-9c8d-97f72146653e)

 - Creating caption by BLIP

 - Create BLIP captions for all images in a folder with "Batch Process" or for a single image with "Single Image".

## License

[MIT](https://choosealicense.com/licenses/mit/)