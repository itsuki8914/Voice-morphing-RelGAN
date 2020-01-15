# Voice-morphing-RelGAN
A implementation of Voice-morphing using RelGAN(image translation) with TensorFlow.

This is under experiment now.

## Original papers
- [CycleGAN-VC2](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc2/index.html)
- [RelGAN](https://arxiv.org/abs/1908.07269)

## Related papers
- [CycleGAN-VC](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc/)
- [StarGAN-VC](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/stargan-vc/)
- [StarGAN-VC2](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/stargan-vc2/index.html)

## Original implementations
- [RelGAN](https://github.com/willylulu/RelGAN)

# Usage
Put the folder containing the wav file in Datasets.
like this

```
...
│
├── train_lr
│     ├── 0001x4.png
│     ├── 0002x4.png
│     ├── ...
│     └── 0800x4.png
├── train_hr
│     ├── 0001.png
│     ├── 0002.png
│     ├── ...
│     └── 0800.png 
├── val_lr
│     ├── 0801x4.png
│     ├── 0802x4.png
│     ├── ...
│     └── 0900x4.png
├── val_hr
│     ├── 0801.png
│     ├── 0802.png
│     ├── ...
│     └── 0900.png 
├── main.py
├── model.py
...
```


 Run preprocess1.py to remove silence　and split the file.
  
```
python preprocess1.py
```


 Run preprocess2.py to extract features and output pickles.
  
```
python preprocess1.py
```

## Acknowledgements
This implementation is based on [njellinas's CycleGAN-VC2](https://github.com/njellinas/GAN-Voice-Conversion).
