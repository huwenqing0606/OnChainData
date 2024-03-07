import subprocess

# Command to install ethereum-etl using pip3
install_command = 'pip3 install ethereum-etl'

# Command to export blocks and transactions: Choose block numbers to export
export_command = 'ethereumetl export_blocks_and_transactions --start-block 000000 --end-block 500000 ' \
                 '--blocks-output blocks3.csv --transactions-output transactions3.csv ' \
                 '--provider-uri https://mainnet.infura.io/v3/7aef3f0cd1f64408b163814b22cc643c'

# Step 1: Install ethereum-etl using pip
try:
    subprocess.run(install_command, shell=True, check=True)
    print("ethereum-etl installed successfully.")
except subprocess.CalledProcessError as e:
    print("Error installing ethereum-etl:", e)
except Exception as e:
    print("An unexpected error occurred:", e)

# Step 2: Export blocks and transactions
try:
    subprocess.run(export_command, shell=True, check=True)
    print("Export completed successfully.")
except subprocess.CalledProcessError as e:
    print("Error exporting blocks and transactions:", e)
except Exception as e:
    print("An unexpected error occurred:", e)
