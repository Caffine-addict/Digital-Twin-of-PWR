"""NPPAD dataset parser (Nuclear Power Plant Accident Dataset).

Integrates the NPPAD.rar dataset containing real PWR transient data.
This parser extracts MDB files and converts them to the system's CSV schema.

Dataset structure:
- 2434 .mdb files (Microsoft Access databases)
- 1217 transient report text files (event logs)
- ~8MB per .mdb file
- Data format: Time-series sensor readings
- Transient types: ATWS, FLB (Anticipated Transient Without Scram, Feed Line Break)

Dependencies:
- mdbtools (system package)
- unixodbc (system package)
- rarfile (Python package)
- pyodbc (optional, for direct ODBC access)

Usage:
    python -m npnad_parser extract --rar-path /path/to/NPPAD.rar --output data/
    python -m npnad_parser convert --input data/DATA/ATWS/1.mdb --output data/real_data.csv
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rarfile

# Mapping of NPPAD variables to our schema
NPPAD_VARIABLE_MAP = {
    "TIME": "time",
    "TAVG": "temperature",  # RCS average temperature
    "PSGA": "pressure",     # Steam generator A pressure (representative)
    "WRCA": "flow_rate",    # Reactor coolant loop A flow
}

# All available NPPAD variables for reference
NPPAD_ALL_VARIABLES = [
    "TIME", "TAVG", "THA", "THB", "TCA", "TCB",
    "WRCA", "WRCB", "PSGA", "PSGB", "WFWA", "WFWB",
    "WSTA", "WSTB", "VOL", "LVPZ", "VOID", "WLR", "WUP",
    # ... [truncated for brevity - full list in module]
]


class NPPADParser:
    """Parser for NPPAD dataset MDB files."""

    def __init__(self, rar_path: Path):
        """Initialize parser with path to NPPAD.rar file."""
        self.rar_path = Path(rar_path)
        if not self.rar_path.exists():
            raise FileNotFoundError(f"NPPAD.rar not found at: {rar_path}")
        self.rar = rarfile.RarFile(self.rar_path)

    def list_transients(self) -> List[str]:
        """List all available transient scenarios."""
        transients = set()
        for name in self.rar.namelist():
            parts = name.split('/')
            if len(parts) >= 2:
                transients.add(parts[1])
        return sorted(list(transients))

    def extract_mdb(self, mdb_path_in_rar: str, output_dir: Path) -> Path:
        """Extract a single MDB file from the RAR archive."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.rar.extract(mdb_path_in_rar, path=output_dir)
        return output_dir / mdb_path_in_rar

    def convert_mdb_to_csv(
        self, 
        mdb_file: Path, 
        output_csv: Path,
        variable_map: Optional[Dict[str, str]] = None
    ) -> Tuple[int, List[str]]:
        """Convert MDB file to CSV using mdb-export.
        
        Returns:
            Tuple of (row_count, list_of_columns)
        """
        if variable_map is None:
            variable_map = NPPAD_VARIABLE_MAP
        
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get column names first
            result = subprocess.run(
                ['mdb-export', '--no-header', str(mdb_file), 'ListPlotVariables'],
                capture_output=True, text=True, check=True
            )
            
            reader = csv.reader(result.stdout.splitlines(), delimiter=',')
            column_mapping = {}
            all_columns = []
            
            for row in reader:
                if len(row) >= 4:
                    var_id, label, units, name = row
                    all_columns.append(name.strip('"'))
                    if name.strip('"') in variable_map:
                        column_mapping[name.strip('"')] = variable_map[name.strip('"')]
            
            # Extract data from PlotData table
            result = subprocess.run(
                ['mdb-export', '--delimiter=,', str(mdb_file), 'PlotData'],
                capture_output=True, text=True, check=True
            )
            
            reader = csv.reader(result.stdout.splitlines(), delimiter=',')
            
            # Find indices of columns we care about
            header = next(reader)
            header = [h.strip('"') for h in header]
            
            target_indices = {}
            for np_var, our_var in column_mapping.items():
                if np_var in header:
                    target_indices[header.index(np_var)] = our_var
            
            # Write to CSV
            with open(output_csv, 'w', newline='') as csvfile:
                fieldnames = list(variable_map.values())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                row_count = 0
                for row in reader:
                    if len(row) < len(header):
                        continue
                        
                    output_row = {}
                    for idx, var_name in target_indices.items():
                        try:
                            output_row[var_name] = float(row[idx])
                        except (ValueError, IndexError):
                            output_row[var_name] = ''
                    
                    writer.writerow(output_row)
                    row_count += 1
            
            return row_count, fieldnames
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to convert MDB file: {e}")
        except FileNotFoundError:
            raise RuntimeError(
                "mdbtools not found. Install with: brew install mdbtools unixodbc"
            )

    def batch_extract_and_convert(
        self,
        output_dir: Path,
        max_files: Optional[int] = None,
        transient_type: Optional[str] = None
    ) -> List[Path]:
        """Extract and convert multiple MDB files.
        
        Args:
            output_dir: Directory to save CSV files
            max_files: Maximum number of files to process (None for all)
            transient_type: Filter by transient type (e.g., 'ATWS', 'FLB')
        
        Returns:
            List of paths to generated CSV files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_files = []
        processed = 0
        
        for name in self.rar.namelist():
            if name.endswith('.mdb'):
                # Filter by transient type if specified
                if transient_type:
                    parts = name.split('/')
                    if len(parts) < 2 or parts[1] != transient_type:
                        continue
                
                print(f"Processing {name}...")
                
                try:
                    # Extract MDB
                    mdb_file = self.extract_mdb(name, output_dir / "temp")
                    
                    # Convert to CSV
                    csv_name = Path(name).stem + "_nppad.csv"
                    csv_path = output_dir / "csv" / csv_name
                    
                    row_count, columns = self.convert_mdb_to_csv(
                        mdb_file, csv_path
                    )
                    
                    print(f"  → {row_count} rows, columns: {columns}")
                    csv_files.append(csv_path)
                    
                    processed += 1
                    if max_files and processed >= max_files:
                        break
                        
                except Exception as e:
                    print(f"  Error: {e}")
                    continue
        
        print(f"\nConverted {len(csv_files)} files to {output_dir}/csv/")
        return csv_files


def convert_mdb_to_csv_standalone(
    mdb_file: Path, 
    output_csv: Path,
    variable_map: Optional[Dict[str, str]] = None
) -> Tuple[int, List[str]]:
    """Standalone MDB to CSV converter that doesn't need RAR."""
    if variable_map is None:
        variable_map = NPPAD_VARIABLE_MAP
    
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get column names first
        result = subprocess.run(
            ['mdb-export', '--no-header', str(mdb_file), 'ListPlotVariables'],
            capture_output=True, text=True, check=True
        )
        
        reader = csv.reader(result.stdout.splitlines(), delimiter=',')
        column_mapping = {}
        all_columns = []
        
        for row in reader:
            if len(row) >= 4:
                var_id, label, units, name = row
                all_columns.append(name.strip('"'))
                if name.strip('"') in variable_map:
                    column_mapping[name.strip('"')] = variable_map[name.strip('"')]
        
        # Extract data from PlotData table
        result = subprocess.run(
            ['mdb-export', '--delimiter=,', str(mdb_file), 'PlotData'],
            capture_output=True, text=True, check=True
        )
        
        reader = csv.reader(result.stdout.splitlines(), delimiter=',')
        
        # Find indices of columns we care about
        header = next(reader)
        header = [h.strip('"') for h in header]
        
        target_indices = {}
        for np_var, our_var in column_mapping.items():
            if np_var in header:
                target_indices[header.index(np_var)] = our_var
        
        # Write to CSV
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = list(variable_map.values())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            row_count = 0
            for row in reader:
                if len(row) < len(header):
                    continue
                    
                output_row = {}
                for idx, var_name in target_indices.items():
                    try:
                        output_row[var_name] = float(row[idx])
                    except (ValueError, IndexError):
                        output_row[var_name] = ''
                
                writer.writerow(output_row)
                row_count += 1
        
        return row_count, fieldnames
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert MDB file: {e}")
    except FileNotFoundError:
        raise RuntimeError(
            "mdbtools not found. Install with: brew install mdbtools unixodbc"
        )


def main():
    parser = argparse.ArgumentParser(description="NPPAD Dataset Parser")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available transients")
    list_parser.add_argument("--rar-path", required=True, help="Path to NPPAD.rar")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract MDB files")
    extract_parser.add_argument("--rar-path", required=True, help="Path to NPPAD.rar")
    extract_parser.add_argument("--output", required=True, help="Output directory")
    extract_parser.add_argument("--max-files", type=int, help="Max files to process")
    extract_parser.add_argument("--transient-type", help="Filter by type (ATWS/FLB)")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert single MDB to CSV")
    convert_parser.add_argument("--input", required=True, help="Input MDB file")
    convert_parser.add_argument("--output", required=True, help="Output CSV file")
    
    args = parser.parse_args()
    
    if args.command == "list":
        parser = NPPADParser(Path(args.rar_path))
        transients = parser.list_transients()
        print("Available transient types:")
        for t in transients:
            print(f"  - {t}")
    
    elif args.command == "extract":
        parser = NPPADParser(Path(args.rar_path))
        parser.batch_extract_and_convert(
            Path(args.output),
            max_files=args.max_files,
            transient_type=args.transient_type
        )
    
    elif args.command == "convert":
        row_count, columns = convert_mdb_to_csv_standalone(
            Path(args.input), Path(args.output)
        )
        print(f"Converted {row_count} rows to {args.output}")
        print(f"Columns: {columns}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
