# GenSeg-HEp2 — GAN-based Data Augmentation for Low-Data HEp-2 Cell Segmentation

Pipeline per la segmentazione di specimen HEp-2 (immunofluorescenza) in regime di dati scarsi. Un generatore pix2pix, la cui architettura è ricercata tramite NAS differenziabile bilivello (libreria `betty-ml`), genera coppie immagine-maschera sintetiche usate per aumentare l'addestramento di una UNet di segmentazione. È incluso un baseline (UNet senza augmentation) per il confronto.

## Requisiti

Python 3.9, PyTorch 1.13.1, CUDA 11.6.

```bash
bash env.sh
conda activate GenSeg
```

## Struttura del dataset

```
data/HEp-2_specimen/
├── train/
│   ├── 00001_p0.tif
│   ├── 00001_p0_Mask.tif
│   └── ...
├── test/
│   ├── 00001_p0_Mask.tif
│   └── ...
├── train.csv   # ID paziente, intensity
└── test.csv
```

## Preparazione dati

1. Crea le patch (256x256, griglia fissa 6x5) da train e test:
```bash
python util/create_patches.py
```
Il file usa path hardcoded (`images_folder`, `patches_folder`); per il test set adattare lo script o duplicarlo puntando a `data/HEp-2_specimen/test`.

2. Crea gli split di cross-validation (5 fold, stratificati per intensity/paziente):
```bash
python util/create_fold_splits.py
```
Genera `train_folds.csv`, richiesto da `util/HEp2_loader.py`.

3. Crea l'indice delle patch di test:
```bash
python util/create_test_indices.py
```
Genera `test_indices.csv`, richiesto per la ricostruzione della maschera intera in fase di test.

## Training e testing

Tutti gli script sotto eseguono 5 fold in subprocess separati (`running_files/*_fold.py`) e aggregano le metriche a fine esecuzione.

```bash
# 1. Pre-training del generatore/discriminatore pix2pix
bash scripts/train_pix2pix_hep2.sh

# 2. Co-training bilivello: NAS del generatore + UNet di segmentazione
bash scripts/train_end2end_hep2.sh

# 3. Baseline: UNet senza augmentation
bash scripts/baseline_hep2.sh

# 4. Inferenza sul test set (Dice score + accuracy)
bash scripts/test_hep2.sh
```

Lo step 2 richiede i pesi pix2pix salvati dallo step 1 in `./pix2pix_HEp2_model/pix2pix-HEp2-fold{N}`.
Lo step 4 legge il modello da `--model_dir` (impostare al path del run baseline o end2end che si vuole valutare).

## Output

```
pix2pix_HEp2_model/pix2pix-HEp2-fold{N}/       # pesi G/D pix2pix per fold
end2end_HEp2_model/end2end-HEp2-unet-fold{N}/  # pesi UNet (best + final) per fold
plots/{pix2pix,end2end,baseline}/              # curve di loss
visuals/{pix2pix,end2end}/                     # confronti immagini reali/generate
test_HEp2/test-HEp2-fold{N}/                   # metriche e visualizzazioni di test
```

## Architettura NAS del generatore

`models_pix2pix/networks.py` (`MixedOp_conv`/`MixedOp_upconv`) sostituisce ogni convoluzione della UNet-generator con una combinazione pesata (softmax) delle primitive definite in `architecture_pix2pix/genotypes.py` (kernel 4x2x1, 6x2x2, 8x2x3). I pesi architetturali (`conv_arch`, `upconv_arch`) sono ottimizzati come parametro separato nel loop bilivello (problema `Arch` in `train_end2end_hep2_fold.py`), non fissati a priori.

## Struttura repository

```
running_files/   # entrypoint + fold worker per pix2pix, end2end, baseline, test
scripts/         # wrapper bash con gli iperparametri
options/         # parsing CLI (base_options.py, train_options.py)
models_pix2pix/  # pix2pix_model.py, base_model.py, networks.py
architecture_pix2pix/  # primitive e genotipi NAS
unet/            # modello di segmentazione
util/HEp2_loader.py, dice_score.py, util.py, create_patches.py,
     create_fold_splits.py, create_test_indices.py
cuda.py          # check ambiente CUDA (diagnostica manuale)
check_masks.py   # QC delle maschere generate da create_patches.py
```
