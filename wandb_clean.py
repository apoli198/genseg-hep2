import wandb
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def delete_all_runs():
    """Delete all runs from your wandb project"""
    try:
        api = wandb.Api()
        runs = api.runs("anony-mouse-557058310966732690/pix2pix_train")
        
        deleted_count = 0
        for run in runs:
            logger.info(f"Deleting run: {run.name}")
            run.delete()
            deleted_count += 1
            
        logger.info(f"Successfully deleted {deleted_count} runs")
        
    except Exception as e:
        logger.error(f"Error deleting runs: {str(e)}")
        logger.error("Stack trace:", exc_info=True)

if __name__ == "__main__":
    delete_all_runs()