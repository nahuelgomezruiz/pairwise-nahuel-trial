"""Logger for component prompts to separate Google Sheets worksheet."""

import logging
from typing import Dict, List
import gspread

logger = logging.getLogger(__name__)


class ComponentPromptLogger:
    """Logs component prompts to a separate worksheet for detailed inspection."""
    
    def __init__(self, sheets_client):
        """Initialize with a sheets client."""
        self.sheets_client = sheets_client
    
    def log_prompts_to_sheet(self, 
                           component_prompts: Dict[str, str],
                           categories: List,
                           spreadsheet_id: str, 
                           run_id: str) -> bool:
        """
        Log component prompts to a separate worksheet.
        
        Args:
            component_prompts: Dict mapping category names to prompts
            categories: List of RubricCategory objects
            spreadsheet_id: Google Sheets spreadsheet ID
            run_id: Run identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Open the spreadsheet
            spreadsheet = self.sheets_client.client.open_by_key(spreadsheet_id)
            
            # Create prompts worksheet name
            prompts_worksheet_name = f"prompts-{run_id.replace(':', '')}"
            
            # Try to get existing worksheet or create new one
            try:
                worksheet = spreadsheet.worksheet(prompts_worksheet_name)
            except gspread.WorksheetNotFound:
                worksheet = spreadsheet.add_worksheet(
                    title=prompts_worksheet_name,
                    rows=len(component_prompts) * 10 + 20,
                    cols=5
                )
                logger.info(f"Created prompts worksheet: {prompts_worksheet_name}")
            
            # Add headers
            headers = [
                "Run ID",
                "Category", 
                "Description",
                "Score Descriptors",
                "Generated Prompt"
            ]
            worksheet.update('A1:E1', [headers])
            
            # Format headers (bold)
            worksheet.format('A1:E1', {
                "textFormat": {"bold": True},
                "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}
            })
            
            # Prepare data
            data_to_write = []
            
            for category in categories:
                category_name = category.name
                
                # Format score descriptors
                score_desc_parts = []
                for score, desc in sorted(category.score_descriptors.items()):
                    score_desc_parts.append(f"Score {score}: {desc}")
                score_descriptors_str = " | ".join(score_desc_parts)
                
                # Get the prompt for this category
                prompt = component_prompts.get(category_name, "No prompt generated")
                
                row_data = [
                    run_id,
                    category_name,
                    category.description,
                    score_descriptors_str,
                    prompt
                ]
                data_to_write.append(row_data)
            
            # Write data
            if data_to_write:
                range_name = f"A2:E{len(data_to_write) + 1}"
                worksheet.update(range_name, data_to_write)
                
                # Note: Column width adjustments require batch update API
                # For now, just log success without column formatting
                
                logger.info(f"Successfully logged {len(data_to_write)} component prompts to {prompts_worksheet_name}")
                return True
            
        except Exception as e:
            logger.error(f"Error logging prompts to sheet: {e}")
            return False
        
        return False