"""
Data cleaning script for DBAASP antimicrobial peptide database.

This script cleans the Verified_AMPs_Database.fasta file by:
1. Removing sequences with non-standard amino acids (X, O, numbers, special chars)
2. Converting lowercase letters to uppercase (D-amino acids -> L-amino acids)
3. Keeping only sequences with standard 20 amino acids

Author: Auto-generated
Date: 2026-01-29
"""

import os
from collections import Counter

# Standard 20 amino acids
STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')


def parse_fasta(filepath):
    """Parse a FASTA file and yield (header, sequence) tuples."""
    sequences = []
    current_header = None
    current_seq = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append((current_header, ''.join(current_seq)))
                current_header = line[1:].strip()  # Remove '>' and strip
                current_seq = []
            else:
                current_seq.append(line)
        
        # Don't forget the last sequence
        if current_header is not None:
            sequences.append((current_header, ''.join(current_seq)))
    
    return sequences


def analyze_sequence(seq):
    """Analyze a sequence and return info about non-standard characters."""
    seq_upper = seq.upper()
    non_standard = []
    lowercase_count = 0
    
    for i, char in enumerate(seq):
        if char.islower():
            lowercase_count += 1
        upper_char = char.upper()
        if upper_char not in STANDARD_AA:
            non_standard.append((i, char))
    
    return {
        'has_lowercase': lowercase_count > 0,
        'lowercase_count': lowercase_count,
        'non_standard_chars': non_standard,
        'is_valid_after_uppercase': all(c in STANDARD_AA for c in seq_upper)
    }


def clean_sequence(seq, mode='strict'):
    """
    Clean a sequence based on the specified mode.
    
    Modes:
    - 'strict': Only keep sequences with standard AA (after uppercase conversion)
    - 'uppercase_only': Convert to uppercase, keep if all standard AA
    - 'remove_x': Remove X characters and keep if remaining is valid
    
    Returns: (cleaned_seq, is_valid) tuple
    """
    # Convert to uppercase
    seq_upper = seq.upper()
    
    if mode == 'strict' or mode == 'uppercase_only':
        # Check if all characters are standard amino acids
        is_valid = all(c in STANDARD_AA for c in seq_upper)
        return seq_upper, is_valid
    
    elif mode == 'remove_x':
        # Remove X characters
        seq_cleaned = ''.join(c for c in seq_upper if c != 'X')
        is_valid = len(seq_cleaned) > 0 and all(c in STANDARD_AA for c in seq_cleaned)
        return seq_cleaned, is_valid
    
    return seq_upper, False


def clean_fasta(input_path, output_path, mode='strict', min_length=5, max_length=200, 
                report_path=None):
    """
    Clean a FASTA file containing AMP sequences.
    
    Parameters:
    - input_path: Path to input FASTA file
    - output_path: Path to output cleaned FASTA file
    - mode: Cleaning mode ('strict', 'uppercase_only', 'remove_x')
    - min_length: Minimum sequence length to keep
    - max_length: Maximum sequence length to keep
    - report_path: Optional path to save cleaning report
    """
    sequences = parse_fasta(input_path)
    
    # Statistics
    stats = {
        'total_input': len(sequences),
        'kept': 0,
        'removed_non_standard': 0,
        'removed_too_short': 0,
        'removed_too_long': 0,
        'removed_empty': 0,
        'had_lowercase': 0,
        'non_standard_chars': Counter()
    }
    
    cleaned_sequences = []
    removed_examples = []
    
    for header, seq in sequences:
        # Skip empty sequences
        if not seq or len(seq.strip()) == 0:
            stats['removed_empty'] += 1
            continue
        
        # Analyze sequence
        analysis = analyze_sequence(seq)
        
        if analysis['has_lowercase']:
            stats['had_lowercase'] += 1
        
        for _, char in analysis['non_standard_chars']:
            stats['non_standard_chars'][char.upper()] += 1
        
        # Clean sequence
        cleaned_seq, is_valid = clean_sequence(seq, mode=mode)
        
        if not is_valid:
            stats['removed_non_standard'] += 1
            if len(removed_examples) < 10:
                removed_examples.append((header, seq, 'non_standard'))
            continue
        
        # Check length constraints
        if len(cleaned_seq) < min_length:
            stats['removed_too_short'] += 1
            if len(removed_examples) < 20:
                removed_examples.append((header, seq, f'too_short ({len(cleaned_seq)})'))
            continue
        
        if len(cleaned_seq) > max_length:
            stats['removed_too_long'] += 1
            continue
        
        # Keep this sequence
        stats['kept'] += 1
        cleaned_sequences.append((header, cleaned_seq))
    
    # Write output
    with open(output_path, 'w') as f:
        for header, seq in cleaned_sequences:
            f.write(f'>{header}\n')
            f.write(f'{seq}\n')
    
    # Print statistics
    print("=" * 60)
    print("FASTA Cleaning Report")
    print("=" * 60)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Mode: {mode}")
    print(f"Length filter: {min_length} - {max_length} aa")
    print("-" * 60)
    print(f"Total input sequences: {stats['total_input']}")
    print(f"Sequences kept: {stats['kept']}")
    print(f"Removed (non-standard AA): {stats['removed_non_standard']}")
    print(f"Removed (too short): {stats['removed_too_short']}")
    print(f"Removed (too long): {stats['removed_too_long']}")
    print(f"Removed (empty): {stats['removed_empty']}")
    print(f"Had lowercase (D-amino acids): {stats['had_lowercase']}")
    print("-" * 60)
    print("Non-standard characters found:")
    for char, count in stats['non_standard_chars'].most_common(20):
        print(f"  '{char}': {count}")
    print("-" * 60)
    print("Examples of removed sequences:")
    for header, seq, reason in removed_examples[:10]:
        print(f"  {header[:40]}... | {seq[:30]}... | Reason: {reason}")
    print("=" * 60)
    
    # Save report if requested
    if report_path:
        with open(report_path, 'w') as f:
            f.write("FASTA Cleaning Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Input file: {input_path}\n")
            f.write(f"Output file: {output_path}\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Length filter: {min_length} - {max_length} aa\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total input sequences: {stats['total_input']}\n")
            f.write(f"Sequences kept: {stats['kept']}\n")
            f.write(f"Removed (non-standard AA): {stats['removed_non_standard']}\n")
            f.write(f"Removed (too short): {stats['removed_too_short']}\n")
            f.write(f"Removed (too long): {stats['removed_too_long']}\n")
            f.write(f"Removed (empty): {stats['removed_empty']}\n")
            f.write(f"Had lowercase (D-amino acids): {stats['had_lowercase']}\n")
    
    return stats, cleaned_sequences


def clean_database(input_name, output_suffix="_cleaned"):
    """Clean a single FASTA database file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, input_name)
    
    # Generate output filename
    base_name = os.path.splitext(input_name)[0]
    output_file = os.path.join(script_dir, f"{base_name}{output_suffix}.fasta")
    report_file = os.path.join(script_dir, f"{base_name}_cleaning_report.txt")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return None
    
    # Clean the FASTA file
    stats, cleaned = clean_fasta(
        input_path=input_file,
        output_path=output_file,
        mode='strict',  # Options: 'strict', 'uppercase_only', 'remove_x'
        min_length=5,   # Minimum peptide length
        max_length=200, # Maximum peptide length (exclude very long sequences)
        report_path=report_file
    )
    
    print(f"\nCleaned FASTA saved to: {output_file}")
    print(f"Report saved to: {report_file}")
    return stats


def main():
    import sys
    
    # Default files to clean
    default_files = [
        "Verified_AMPs_Database.fasta",
        "Verified_AFPs_Database.fasta",
    ]
    
    # If command line arguments provided, use them
    if len(sys.argv) > 1:
        files_to_clean = sys.argv[1:]
    else:
        files_to_clean = default_files
    
    # Clean each file
    for filename in files_to_clean:
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")
        clean_database(filename)


if __name__ == "__main__":
    main()
