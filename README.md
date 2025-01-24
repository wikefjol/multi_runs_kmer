# multi_runs_kmer

## Getting Started

1. **Create a Virtual Environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # On macOS/Linux
   # or:
   # venv\Scripts\activate.bat # On Windows
   ```

2. **Install Requirements**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Add Your Data**  
   - Place your `raw.fasta` inside the `data/` folder.

4. **Run the Configs**  
   - Each JSON config file in `configs/` must have a **unique** name.  
   - From the project root, run:
     ```bash
     python run_configs.py
     ```
   - The script will automatically create the `runs/` folder, with subfolders matching the names of each config file.  
   - You’ll find logs, plots, and other outputs in those subfolders.
   - **If you change things in the configs and want to rerun the run_configs**, it might be a good idea to delete the runs folder before compiling.  

**Note**: The repository is set up so that each configuration is handled independently, producing its own run folder and results. Make sure any new config files you add also have unique filenames.