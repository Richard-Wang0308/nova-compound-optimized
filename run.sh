export CUDA_VISIBLE_DEVICES=0,1    # make both GPUs visible
python3 neurons/miner/miner_optimized.py --wallet.name multisig-jjpes-atel --wallet.hotkey hotb --logging.info
