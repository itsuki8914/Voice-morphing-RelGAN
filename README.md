# Voice-morphing-RelGAN
A implementation of Voice-morphing using RelGAN(image translation) with TensorFlow.

This enables Many to many voice conversion and voice morphing.

This is under experiment now.

## Original papers and pages
- [CycleGAN-VC2](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc2/index.html)
- [RelGAN](https://arxiv.org/abs/1908.07269)

## Related papers and pages
- [CycleGAN-VC](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc/)
- [StarGAN-VC](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/stargan-vc/)
- [StarGAN-VC2](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/stargan-vc2/index.html)

## Original implementations
- [RelGAN](https://github.com/willylulu/RelGAN)

# Usage
1. Put the folder containing the wav files for training in named datasets.

  Folders are needed 3 or more.

  And Put the folder containing a few wav files for validation in datasets_val.
 
  like this

```
...
│
datasets
|   │
|   ├── speaker_1
|   │     ├── wav1_1.wav
|   │     ├── wav1_2.wav
|   │     ├── ...
|   │     └── wav1_i.wav
|   ├── speaker_2
|   │     ├── wav2_1.wav
|   │     ├── wav2_2.wav
|   │     ├── ...
|   │     └── wav2_j.wav 
|   ...
|   └── speaker_N
|         ├── wavN_1.wav
|         ├── wavN_2.wav
|         ├── ...
|         └── wavN_k.wav    
datasets_val
|   │
|   ├── speaker_1
|   │     ├── wav1_i+1.wav
|   │     ├── wav1_i+2.wav
|   │     ├── ...
|   │     └── wav1_i+5.wav
|   ├── speaker_2
|   │     ├── wav2_j+1.wav
|   │     ├── wav2_j+2.wav
|   │     ├── ...
|   │     └── wav2_j+3.wav 
|   ...
|   └── speaker_N
|         ├── wavN_k+1.wav
|         ├── wavN_k+2.wav
|         ├── ...
|         └── wavN_k+4.wav 
...
├── preprocess1.py     
├── preprocess2.py
...
```

2. Run preprocess1.py to remove silence and split the file.
  
```
python preprocess1.py
```

3. Run preprocess2.py to extract features and output pickles.
  
```
python preprocess2.py
```

4. Train RelGAN-VM.

```
python train_relgan_vm.py
```

5. After training, inference can be performed.

   Source attribute and target attribute must be designated.
   
   In below example, The 2nd attribute wav file, datasets_val/speaker_2, will be 60% converted to the 4th attribute (probably speaker_4).
   
   pay attention to 0-origin index.

```
python eval_relgan_vm.py --source_label 1 --source_label 3 --interpolation 0.6
```

## Result examples
The examples trained using [JVS (Japanese versatile speech) corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus) are located in result_examples.

The following four voices were used for training.

* jvs010(female, high-pitched fo, domain 0)
* jvs016(female, low-pitched fo, domain 1)
* jvs042(male, low-pitched fo, domain 2)
* jvs054(male, high-pitched fo, domain 3)

## Acknowledgements
This implementation is based on [njellinas's CycleGAN-VC2](https://github.com/njellinas/GAN-Voice-Conversion).

And this was created with the advice of [Lgeu](https://twitter.com/lgeuwce).
