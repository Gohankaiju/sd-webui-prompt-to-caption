# sd-webui-prompt-to-caption
 - [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 用のキャプショニング拡張
 - AIによって生成された画像を用いてLoRA等のファインチューニングを行う場合に有用です。

  [<img src="https://img.shields.io/badge/lang-Egnlish-red.svg?style=plastic" height="25" />](README.md)
 [<img src="https://img.shields.io/badge/言語-日本語-green.svg?style=plastic" height="25" />](#overview)

## Introduction
 - Stable Diffusion等によって生成された画像からプロンプト情報を取り出し、キャプションテキストファイルを生成します。

 - LoRA情報は自動で削除され、プロンプトのみをキャプションとしてテキストファイルを作成します。

 - 加筆、トリミングなどによってメタデータを取得できなかった場合、代わりにBLIPによるキャプショニングを行うこともできます。

## How to Use
### Prompt
![2](https://github.com/Gohankaiju/sd-webui-prompt-to-caption/assets/167270541/99ed0d60-e54e-4d02-972b-d5ec41ec284e)

 - 画像ファイルのフォルダパスを入力し、Genボタンを押します。

 - "Enable BLIP" ボックスにチェックを入れることで、データが取得できなかった画像に対して、代わりにBLIPでキャプションを生成します。(チェックを入れない場合、データ取得ができなかった画像には空のキャプションファイルが作成されます。)

### BLIP
![1](https://github.com/Gohankaiju/sd-webui-prompt-to-caption/assets/167270541/9a98f94c-d465-477a-9c8d-97f72146653e)

 - おまけとしてBLIPによるキャプション生成が可能です。

 - "Batch Process" によってフォルダの画像全てにキャプションファイルを作成するか、"Single image" によって一枚の画像に対してキャプションを表示することができます。

## License

[MIT](https://choosealicense.com/licenses/mit/)