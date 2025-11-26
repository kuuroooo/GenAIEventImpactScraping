import subprocess
import sys
import time
import os

# ==========================================
# VISUAL CONFIGURATION
# ==========================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_step(step_name):
    print(f"\n{Colors.HEADER}========================================{Colors.ENDC}")
    print(f"{Colors.BOLD}>> EXECUTING PHASE: {step_name}{Colors.ENDC}")
    print(f"{Colors.HEADER}========================================{Colors.ENDC}")

def run_command(command):
    """Runs a shell command and streams output to terminal"""
    try:
        # python -u unbuffers output so it prints in real-time
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')

        process.wait()
        
        if process.returncode != 0:
            print(f"\n{Colors.FAIL}[CRITICAL] Step failed with exit code {process.returncode}{Colors.ENDC}")
            return False
        return True

    except Exception as e:
        print(f"{Colors.FAIL}[ERROR] Execution failed: {e}{Colors.ENDC}")
        return False

# ==========================================
# MAIN PIPELINE
# ==========================================

def main():
    start_time = time.time()
    
    print(f"""{Colors.CYAN}
   _____  _____ _____  ______ _      _____ _   _ ______ 
  |  __ \|_   _|  __ \|  ____| |    |_   _| \ | |  ____|
  | |__) | | | | |__) | |__  | |      | | |  \| | |__   
  |  ___/  | | |  ___/|  __| | |      | | | . ` |  __|  
  | |     _| |_| |    | |____| |____ _| |_| |\  | |____ 
  |_|    |_____|_|    |______|______|_____|_| \_|______|
                                                        
    {Colors.GREEN}:: AUTOMATED INTELLIGENCE GATHERING ::{Colors.ENDC}
    """)

    # --- STEP 1: SCRAPE ---
    print_step("1. RAW DATA COLLECTION")
    # You can customize arguments here (e.g., --days 3)
    if not run_command(f"{sys.executable} -u scrape_raw.py"):
        sys.exit(1)

    # --- STEP 2: ANALYZE ---
    print_step("2. SENTIMENT PROCESSING")
    if not run_command(f"{sys.executable} -u sandy.py"):
        sys.exit(1)

    # --- STEP 3: VISUALIZE ---
    print_step("3. VISUALIZATION GENERATION")
    if not run_command(f"{sys.executable} -u analyze.py"):
        sys.exit(1)

    # --- SUMMARY ---
    elapsed = time.time() - start_time
    print(f"\n{Colors.GREEN}✔ PIPELINE COMPLETE{Colors.ENDC}")
    print(f"  Total Runtime: {elapsed:.2f} seconds")
    print(f"  Outputs: {Colors.BOLD}analysis_timeline.png{Colors.ENDC}, {Colors.BOLD}analysis_platforms.png{Colors.ENDC}")

if __name__ == "__main__":
    main()