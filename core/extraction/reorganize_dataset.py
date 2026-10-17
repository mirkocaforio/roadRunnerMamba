#!/usr/bin/env python3
"""
Script per riorganizzare il dataset RSCD secondo la struttura richiesta:
- Estrae la prima classe dal nome file (es: dry, wet, ecc.)
- Crea sottocartelle per classe in train, val e test
- Sposta le immagini nelle rispettive cartelle
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm

def extract_first_class(filename):
    """
    Estrae la prima classe dal nome file.
    Esempio: '202201261342122-dry-asphalt-severe.jpg' -> 'dry'
    """
    # Rimuovi estensione
    name_without_ext = Path(filename).stem
    
    # Split per '-' e prendi il secondo elemento (dopo timestamp)
    parts = name_without_ext.split('-')
    if len(parts) < 2:
        return None
    
    # La prima classe è dopo il timestamp
    first_class = parts[1].lower().strip()
    return first_class

def scan_images(folder_path):
    """
    Scansiona una cartella e raggruppa le immagini per classe.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    class_images = defaultdict(list)
    
    folder = Path(folder_path)
    
    # Se la cartella ha sottocartelle (come train), scansiona ricorsivamente
    if any(p.is_dir() for p in folder.iterdir()):
        for subfolder in folder.iterdir():
            if subfolder.is_dir():
                for img_file in subfolder.iterdir():
                    if img_file.suffix.lower() in image_extensions:
                        first_class = extract_first_class(img_file.name)
                        if first_class:
                            class_images[first_class].append(img_file)
    else:
        # Cartella piatta (come val/test)
        for img_file in folder.iterdir():
            if img_file.suffix.lower() in image_extensions:
                first_class = extract_first_class(img_file.name)
                if first_class:
                    class_images[first_class].append(img_file)
    
    return class_images

def reorganize_folder(source_folder, dest_folder, dry_run=False):
    """
    Riorganizza una cartella secondo la nuova struttura.
    """
    print(f"\n=== Processing {source_folder.name} ===")
    
    # Scansiona immagini
    class_images = scan_images(source_folder)
    
    if not class_images:
        print(f"WARNING: No images found in {source_folder}")
        return
    
    # Mostra statistiche
    print(f"Found {len(class_images)} classes:")
    for class_name, images in sorted(class_images.items()):
        print(f"  {class_name}: {len(images)} images")
    
    if dry_run:
        print("DRY RUN: No files will be moved")
        return
    
    # Crea cartella destinazione
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    # Crea sottocartelle per ogni classe e sposta immagini
    total_moved = 0
    for class_name, images in tqdm(class_images.items(), desc=f"Reorganizing {source_folder.name}"):
        class_folder = dest_folder / class_name
        class_folder.mkdir(exist_ok=True)
        
        for img_file in images:
            dest_file = class_folder / img_file.name
            
            # Se il file esiste già, aggiungi un suffisso
            if dest_file.exists():
                base = dest_file.stem
                ext = dest_file.suffix
                counter = 1
                while dest_file.exists():
                    dest_file = class_folder / f"{base}_{counter}{ext}"
                    counter += 1
            
            shutil.move(str(img_file), str(dest_file))
            total_moved += 1
    
    print(f"Moved {total_moved} images to {dest_folder}")

def reorganize_dataset(dataset_path, output_path=None, dry_run=False, backup=True):
    """
    Riorganizza l'intero dataset.
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path not found: {dataset_path}")
    
    # Se output_path non specificato, usa dataset_path_reorganized
    if output_path is None:
        output_path = dataset_path.parent / f"{dataset_path.name}_reorganized"
    else:
        output_path = Path(output_path)
    
    print(f"=== RSCD Dataset Reorganization ===")
    print(f"Source: {dataset_path}")
    print(f"Output: {output_path}")
    print(f"Dry run: {dry_run}")
    
    # Backup originale se richiesto
    if backup and not dry_run and not output_path.exists():
        print("\nCreating backup of original dataset...")
        backup_path = dataset_path.parent / f"{dataset_path.name}_backup"
        if not backup_path.exists():
            shutil.copytree(dataset_path, backup_path)
            print(f"Backup created at: {backup_path}")
    
    # Processa train, val, test
    for split in ['train', 'val', 'test']:
        source_folder = dataset_path / split
        dest_folder = output_path / split
        
        if not source_folder.exists():
            print(f"WARNING: {split} folder not found, skipping...")
            continue
        
        reorganize_folder(source_folder, dest_folder, dry_run)
    
    # Rimuovi cartelle vuote nella source (se non dry run)
    if not dry_run:
        print("\n=== Cleaning up empty folders ===")
        for split in ['train', 'val', 'test']:
            source_folder = dataset_path / split
            if source_folder.exists():
                for subfolder in source_folder.iterdir():
                    if subfolder.is_dir() and not any(subfolder.iterdir()):
                        print(f"Removing empty folder: {subfolder}")
                        subfolder.rmdir()
    
    print("\n=== Reorganization Complete ===")
    print(f"New dataset structure saved in: {output_path}")
    
    # Mostra struttura finale
    print("\n=== Final Structure ===")
    for split in ['train', 'val', 'test']:
        split_folder = output_path / split
        if split_folder.exists():
            print(f"\n{split}/")
            for class_folder in sorted(split_folder.iterdir()):
                if class_folder.is_dir():
                    num_images = len(list(class_folder.glob('*')))
                    print(f"  {class_folder.name}/  ({num_images} images)")

def main():
    parser = argparse.ArgumentParser(
        description='Reorganize RSCD dataset by first class label',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to preview changes
  python reorganize_dataset.py /path/to/RSCD_dataset --dry-run
  
  # Reorganize with backup
  python reorganize_dataset.py /path/to/RSCD_dataset
  
  # Specify custom output path
  python reorganize_dataset.py /path/to/RSCD_dataset -o /path/to/output
  
  # No backup
  python reorganize_dataset.py /path/to/RSCD_dataset --no-backup
        """
    )
    
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to RSCD dataset root folder'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path for reorganized dataset (default: {dataset_path}_reorganized)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually moving files'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup of original dataset'
    )
    
    args = parser.parse_args()
    
    try:
        reorganize_dataset(
            dataset_path=args.dataset_path,
            output_path=args.output,
            dry_run=args.dry_run,
            backup=not args.no_backup
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())