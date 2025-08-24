"""Google Sheets client for essay scoring integration."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

try:
    import gspread
    from google.oauth2.service_account import Credentials
except ImportError:
    raise ImportError("Please install Google Sheets dependencies: pip install gspread google-auth")

from config.settings import GOOGLE_SHEETS_CREDENTIALS_PATH

logger = logging.getLogger(__name__)


class SheetsClient:
    """Client for Google Sheets integration."""
    
    def __init__(self, credentials_path: str = None, credentials_dict: dict = None):
        """Initialize Google Sheets client."""
        if credentials_path is None and credentials_dict is None:
            credentials_path = GOOGLE_SHEETS_CREDENTIALS_PATH
        
        self.credentials_path = credentials_path
        self.credentials_dict = credentials_dict
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the Google Sheets client with credentials."""
        try:
            # Define the scope for Google Sheets API
            scope = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            # Load credentials
            if self.credentials_dict:
                # Use credentials dictionary (from environment variable)
                credentials = Credentials.from_service_account_info(
                    self.credentials_dict, 
                    scopes=scope
                )
                self.client = gspread.authorize(credentials)
                logger.info("Successfully initialized Google Sheets client from credentials dict")
            elif self.credentials_path and Path(self.credentials_path).exists():
                # Use credentials file
                credentials = Credentials.from_service_account_file(
                    self.credentials_path, 
                    scopes=scope
                )
                self.client = gspread.authorize(credentials)
                logger.info("Successfully initialized Google Sheets client from file")
            else:
                logger.warning(f"No valid credentials provided. File path: {self.credentials_path}")
                self.client = None
                
        except Exception as e:
            logger.error(f"Error initializing Google Sheets client: {e}")
            self.client = None

    def calculate_quadratic_weighted_kappa(self, actual_scores: List[float], predicted_scores: List[float]) -> float:
        """
        Calculate Quadratic Weighted Kappa (QWK) according to specification.
        
        QWK is the standard metric for essay scoring competitions.
        
        Rating scale: 0…N-1 ordered categories
        Oᵢⱼ: observed count of essays with true rating i and predicted rating j
        wᵢⱼ: weight = ((i − j)²)/(N − 1)²
        Eᵢⱼ: expected count under independence = (nᵢ mⱼ)/T
        κ = 1 − (Numerator / Denominator)
        
        Args:
            actual_scores: List of actual/ground truth scores
            predicted_scores: List of predicted scores
            
        Returns:
            Quadratic Weighted Kappa value
        """
        if len(actual_scores) != len(predicted_scores):
            logger.warning("Actual and predicted scores length mismatch")
            return 0.0
            
        if not actual_scores or not predicted_scores:
            return 0.0
            
        try:
            # Convert to numpy arrays and round to integers
            actual = np.round(np.array(actual_scores)).astype(int)
            predicted = np.round(np.array(predicted_scores)).astype(int)
            
            # Determine the full rating scale (should cover all possible ratings)
            min_rating = min(min(actual), min(predicted))
            max_rating = max(max(actual), max(predicted))
            N = max_rating - min_rating + 1  # Number of categories
            
            # Map ratings to 0-based indices for the specification
            actual_idx = actual - min_rating  # Convert to 0…N-1
            predicted_idx = predicted - min_rating  # Convert to 0…N-1
            
            # Build the N × N observed matrix O from all paired scores
            O = np.zeros((N, N))
            for a_idx, p_idx in zip(actual_idx, predicted_idx):
                O[a_idx, p_idx] += 1
            
            # Calculate row totals (nᵢ), column totals (mⱼ), and total essays (T)
            n_i = np.sum(O, axis=1)  # Row totals
            m_j = np.sum(O, axis=0)  # Column totals  
            T = np.sum(O)  # Total essays
            
            # Expected matrix: Eᵢⱼ = (nᵢ mⱼ)/T
            E = np.outer(n_i, m_j) / T
            
            # Create weight matrix: wᵢⱼ = ((i − j)²)/(N − 1)²
            W = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    if N > 1:  # Avoid division by zero
                        W[i, j] = ((i - j) ** 2) / ((N - 1) ** 2)
                    else:
                        W[i, j] = 0  # Single category case
            
            # Calculate QWK components
            numerator = np.sum(W * O)  # ΣᵢΣⱼ wᵢⱼ Oᵢⱼ
            denominator = np.sum(W * E)  # ΣᵢΣⱼ wᵢⱼ Eᵢⱼ
            
            if denominator == 0:
                logger.warning("QWK denominator is zero - returning 0")
                return 0.0
                
            # κ = 1 − (Numerator / Denominator)
            kappa = 1 - (numerator / denominator)
            
            logger.debug(f"QWK calculation: N={N}, T={T}, numerator={numerator:.4f}, denominator={denominator:.4f}, κ={kappa:.4f}")
            return float(kappa)
            
        except Exception as e:
            logger.error(f"Error calculating QWK: {e}")
            return 0.0
     
    def write_scores_to_sheet(self, scores: List[Dict[str, Any]], 
                             spreadsheet_id: str,
                             worksheet_name: str = "Results",
                             start_row: int = 2,
                             create_headers: bool = True,
                             run_id: str = None,
                             actual_scores: List[float] = None,
                             component_categories: List = None) -> bool:
        """
        Write scores to a Google Sheet with component-specific columns.
        
        Args:
            scores: List of score dictionaries to write
            spreadsheet_id: Google Sheets spreadsheet ID
            worksheet_name: Name of the worksheet
            start_row: Row number to start writing from
            create_headers: Whether to create header row
            run_id: Run identifier (hour:minute:second format)
            actual_scores: List of actual scores for QWK calculation
            component_categories: List of RubricCategory objects for individual columns
        
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            raise RuntimeError("Google Sheets client not initialized")
        
        if not scores:
            logger.warning("No scores to write to Google Sheets")
            return False
        
        # Generate run_id if not provided
        if not run_id:
            now = datetime.now()
            run_id = now.strftime("%H:%M:%S")
            
        try:
            # Open the spreadsheet
            spreadsheet = self.client.open_by_key(spreadsheet_id)
            
            # Try to get existing worksheet or create new one
            try:
                worksheet = spreadsheet.worksheet(worksheet_name)
            except gspread.WorksheetNotFound:
                worksheet = spreadsheet.add_worksheet(
                    title=worksheet_name, 
                    rows=len(scores) + 20, 
                    cols=10
                )
                logger.info(f"Created new worksheet: {worksheet_name}")
                
                # Add headers for new worksheet - base headers plus individual component columns
                headers = [
                    "Run ID",
                    "Essay ID", 
                    "AI Score",
                    "Actual Score",
                    "Mean (components)",
                    "Geometric Mean",
                    "Harmonic Mean",
                    "Essay Relevance",  # Essay relevance column
                ]
                
                # Add individual component columns
                if component_categories:
                    for category in component_categories:
                        headers.append(f"{category.name}")
                
                # Add remaining headers
                headers.extend([
                    "Essay Text",
                    "AI Reasoning",
                    "Model",
                    "QWK",
                    "Component Scores",
                    "Component Prompts"
                ])
                
                # Calculate the range based on number of headers
                end_col = chr(ord('A') + len(headers) - 1)
                if len(headers) > 26:
                    # Handle more than 26 columns (AA, AB, etc.)
                    end_col = f"A{chr(ord('A') + len(headers) - 27)}"
                
                worksheet.update(f'A1:{end_col}1', [headers])
                
                # Format headers (bold)
                end_col = chr(ord('A') + len(headers) - 1)
                if len(headers) > 26:
                    end_col = f"A{chr(ord('A') + len(headers) - 27)}"
                
                worksheet.format(f'A1:{end_col}1', {
                    "textFormat": {"bold": True},
                    "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}
                })
                
                logger.info(f"Added headers to new worksheet: {worksheet_name}")
            
            # Calculate QWK if actual scores are provided
            qwk = None
            qwk_display = ""
            if actual_scores and len(actual_scores) > 0:
                predicted_scores = [score["total_score"] for score in scores]
                if len(actual_scores) == len(predicted_scores):
                    qwk = self.calculate_quadratic_weighted_kappa(actual_scores, predicted_scores)
                    qwk_display = f"{qwk:.4f}"
                    logger.info(f"Calculated QWK: {qwk:.4f}")
                else:
                    logger.warning(f"Mismatch in score lengths: {len(actual_scores)} actual vs {len(predicted_scores)} predicted")
            else:
                logger.info("No actual scores provided - QWK will not be calculated")
            
            # Find the next available row to write to
            try:
                # Get all values to find the last used row
                existing_values = worksheet.get_all_values()
                next_row = len(existing_values) + 1 if existing_values else 1
            except:
                next_row = 1
            
            # Prepare data for writing - only essay data rows
            data_to_write = []
            
            # Add score data only
            for i, score in enumerate(scores):
                # Get actual score if available
                actual_score = actual_scores[i] if actual_scores and i < len(actual_scores) else ""
                
                # Get the essay text from the score metadata or original essay data
                essay_text = ""
                if 'essay_text' in score:
                    essay_text = score['essay_text']
                elif 'metadata' in score and score['metadata'] and 'essay_text' in score['metadata']:
                    essay_text = score['metadata']['essay_text']
                
                # Keep full essay text for thorough review
                # Note: Google Sheets can handle up to 50,000 characters per cell
                
                # Start building row data
                row_data = [
                    run_id,
                    score["essay_id"],
                    score["total_score"],
                    actual_score,  # Actual score column
                ]

                # Compute per-essay means across all categories (including Essay Relevance if present)
                mean_components = ""
                geo_mean = ""
                harm_mean = ""
                if "category_scores" in score and score["category_scores"]:
                    try:
                        vals = [float(v) for v in score["category_scores"].values() if v is not None]
                        if len(vals) > 0:
                            mean_components = float(np.mean(vals))
                            # Geometric mean and harmonic mean assume positive values; rubric scores are 1..6
                            arr = np.array(vals, dtype=float)
                            geo_mean = float(np.exp(np.mean(np.log(arr))))
                            harm_mean = float(len(arr) / np.sum(1.0 / arr))
                    except Exception:
                        pass

                row_data.extend([
                    mean_components if mean_components != "" else "",
                    geo_mean if geo_mean != "" else "",
                    harm_mean if harm_mean != "" else "",
                ])
                
                # Add Essay Relevance score (separately, as it's weighted differently)
                essay_relevance_score = ""
                if "category_scores" in score and score["category_scores"]:
                    essay_relevance_score = score["category_scores"].get("Essay Relevance", "")
                row_data.append(essay_relevance_score)
                
                # Add individual component scores (excluding Essay Relevance)
                if component_categories:
                    for category in component_categories:
                        category_score = ""
                        if "category_scores" in score and score["category_scores"]:
                            # Don't duplicate Essay Relevance here
                            if category.name != "Essay Relevance":
                                category_score = score["category_scores"].get(category.name, "")
                        row_data.append(category_score)
                
                # Format component scores summary for backward compatibility
                component_scores_str = ""
                if "category_scores" in score and score["category_scores"]:
                    component_parts = []
                    for category, cat_score in score["category_scores"].items():
                        component_parts.append(f"{category}: {cat_score}")
                    component_scores_str = " | ".join(component_parts)
                
                # Format component prompts if available (do not truncate)
                component_prompts_str = ""
                if "metadata" in score and score["metadata"] and "component_prompts" in score["metadata"]:
                    prompt_parts = []
                    for category, prompt in score["metadata"]["component_prompts"].items():
                        prompt_parts.append(f"{category}: {prompt}")
                    component_prompts_str = " | ".join(prompt_parts)
                
                # Add remaining columns
                row_data.extend([
                    essay_text,    # Raw essay text column
                    score["reasoning"],  # Keep full AI reasoning for thorough review
                    score.get("model_used", ""),
                    "",  # Empty QWK column for individual essays
                    component_scores_str,  # Component scores breakdown
                    component_prompts_str   # Component prompts used
                ])
                
                data_to_write.append(row_data)
            
            # Add aggregated metrics row
            if len(scores) > 0:
                avg_score = np.mean([score["total_score"] for score in scores])
                std_score = np.std([score["total_score"] for score in scores])
                
                # Calculate average actual score if available
                avg_actual = np.mean(actual_scores) if actual_scores else ""
                
                # Start building aggregated row
                aggregated_row = [
                    f"AGG_{run_id}",  # Aggregated run ID
                    "AGGREGATED",
                    f"Avg: {avg_score:.2f}, Std: {std_score:.2f}",
                    f"Avg: {avg_actual:.2f}" if avg_actual else "N/A",  # Average actual score
                ]

                # Aggregated per-essay means (averaged across essays)
                per_essay_means = []
                per_essay_geo_means = []
                per_essay_harm_means = []
                relevance_list = []
                for s in scores:
                    if "category_scores" in s and s["category_scores"]:
                        vals = [float(v) for v in s["category_scores"].values() if v is not None]
                        if len(vals) > 0:
                            arr = np.array(vals, dtype=float)
                            per_essay_means.append(float(np.mean(arr)))
                            per_essay_geo_means.append(float(np.exp(np.mean(np.log(arr)))))
                            per_essay_harm_means.append(float(len(arr) / np.sum(1.0 / arr)))
                        if "Essay Relevance" in s["category_scores"]:
                            relevance_list.append(float(s["category_scores"]["Essay Relevance"]))

                aggregated_row.extend([
                    f"Avg: {np.mean(per_essay_means):.2f}" if per_essay_means else "N/A",
                    f"Avg: {np.mean(per_essay_geo_means):.2f}" if per_essay_geo_means else "N/A",
                    f"Avg: {np.mean(per_essay_harm_means):.2f}" if per_essay_harm_means else "N/A",
                    f"Avg: {np.mean(relevance_list):.2f}" if relevance_list else "",
                ])
                
                # Add component averages
                if component_categories:
                    for category in component_categories:
                        cat_scores = []
                        for score in scores:
                            if 'category_scores' in score and category.name in score['category_scores']:
                                cat_scores.append(score['category_scores'][category.name])
                        
                        if cat_scores:
                            cat_avg = np.mean(cat_scores)
                            aggregated_row.append(f"Avg: {cat_avg:.2f}")
                        else:
                            aggregated_row.append("")
                
                # Add remaining columns for aggregated row
                aggregated_row.extend([
                    f"Contains {len(scores)} essays",  # Essay text column for aggregated row
                    f"Aggregated metrics for {len(scores)} essays",
                    "AGGREGATE",
                    qwk_display if qwk is not None else "N/A",
                    "",  # Empty component scores for aggregate row
                    ""   # Empty component prompts for aggregate row
                ])
                
                data_to_write.append(aggregated_row)
            
            # Write data to next empty rows
            num_cols = len(data_to_write[0]) if data_to_write else 10
            end_col = chr(ord('A') + num_cols - 1)
            if num_cols > 26:
                end_col = f"A{chr(ord('A') + num_cols - 27)}"
            
            range_name = f"A{next_row}:{end_col}{next_row + len(data_to_write) - 1}"
            worksheet.update(range_name, data_to_write)
            
            # Apply conditional formatting to actual score column for essay rows
            try:
                if actual_scores and len(scores) > 0:
                    spreadsheet_obj = self.client.open_by_key(spreadsheet_id)
                    worksheet_id = worksheet.id
                    
                    # Create conditional formatting rules for each essay row
                    # Color coding based on score disagreement magnitude:
                    # Green (≤0.1): Excellent match
                    # Yellow (≤0.5): Good match  
                    # Orange (≤1.0): Moderate disagreement
                    # Dark Orange (≤1.5): Large disagreement
                    # Light Red (≤2.0): Very large disagreement  
                    # Dark Red (>2.0): Extreme disagreement
                    requests = []
                    
                    for i in range(len(scores)):
                        row_num = next_row + i
                        predicted_score = scores[i]["total_score"]
                        actual_score = actual_scores[i] if i < len(actual_scores) else None
                        
                        if actual_score is not None:
                            # Calculate score disagreement magnitude
                            disagreement = abs(predicted_score - actual_score)
                            
                            # Choose color based on disagreement magnitude with heavier colors for larger disagreements
                            if disagreement <= 0.1:
                                # Excellent match - light green
                                bg_color = {"red": 0.85, "green": 1.0, "blue": 0.85}
                            elif disagreement <= 0.5:
                                # Good match - light yellow
                                bg_color = {"red": 1.0, "green": 1.0, "blue": 0.8}
                            elif disagreement <= 1.0:
                                # Moderate disagreement - orange
                                bg_color = {"red": 1.0, "green": 0.8, "blue": 0.4}
                            elif disagreement <= 1.5:
                                # Large disagreement - dark orange  
                                bg_color = {"red": 1.0, "green": 0.6, "blue": 0.2}
                            elif disagreement <= 2.0:
                                # Very large disagreement - light red
                                bg_color = {"red": 1.0, "green": 0.4, "blue": 0.4}
                            else:
                                # Extreme disagreement - dark red
                                bg_color = {"red": 0.8, "green": 0.2, "blue": 0.2}
                            
                            # Log the color coding for debugging
                            logger.debug(f"Essay {scores[i]['essay_id']}: disagreement={disagreement:.2f}, color intensity applied")
                            
                            # Apply formatting to the actual score cell (column D)
                            requests.append({
                                "repeatCell": {
                                    "range": {
                                        "sheetId": worksheet_id,
                                        "startRowIndex": row_num - 1,  # 0-indexed
                                        "endRowIndex": row_num,
                                        "startColumnIndex": 3,  # Column D (actual score)
                                        "endColumnIndex": 4
                                    },
                                    "cell": {
                                        "userEnteredFormat": {
                                            "backgroundColor": bg_color
                                        }
                                    },
                                    "fields": "userEnteredFormat(backgroundColor)"
                                }
                            })
                    
                    if requests:
                        spreadsheet_obj.batch_update({"requests": requests})
                        logger.info(f"Applied conditional formatting to {len(requests)} actual score cells")
                        
            except Exception as e:
                logger.warning(f"Could not apply conditional formatting: {e}")
            
            # Create Excel grouping for this run's essay rows
            try:
                # Group the essay rows AND the aggregated row
                if len(scores) > 1:  # Only group if there are multiple essays
                    group_start = next_row + 1  # Start from second essay row
                    group_end = next_row + len(scores)  # Include aggregated row
                    
                    # Use Google Sheets API to create row grouping
                    spreadsheet_obj = self.client.open_by_key(spreadsheet_id)
                    worksheet_id = worksheet.id
                    
                    requests = [{
                        "addDimensionGroup": {
                            "range": {
                                "sheetId": worksheet_id,
                                "dimension": "ROWS",
                                "startIndex": group_start - 1,  # 0-indexed for API
                                "endIndex": group_end  # 0-indexed for API, exclusive end
                            }
                        }
                    }]
                    
                    spreadsheet_obj.batch_update({"requests": requests})
                    logger.info(f"Created Excel group for run {run_id}: rows {group_start} to {group_end} (includes aggregated row)")
                
            except Exception as e:
                logger.warning(f"Could not create Excel grouping for run {run_id}: {e}")
            
            # Format the aggregated row (if possible)
            try:
                # The aggregated row is the last row we just wrote
                aggregated_row_num = next_row + len(scores)  # After all essay rows
                
                logger.info(f"Aggregated metrics row at line {aggregated_row_num}")
                
                # Apply formatting to aggregated row
                spreadsheet_obj = self.client.open_by_key(spreadsheet_id)
                worksheet_id = worksheet.id
                requests = [{
                    "repeatCell": {
                        "range": {
                            "sheetId": worksheet_id,
                            "startRowIndex": aggregated_row_num - 1,  # 0-indexed
                            "endRowIndex": aggregated_row_num,
                            "startColumnIndex": 0,
                            "endColumnIndex": num_cols  # Dynamic based on number of columns
                        },
                        "cell": {
                            "userEnteredFormat": {
                                "backgroundColor": {
                                    "red": 0.9,
                                    "green": 0.9,
                                    "blue": 1.0
                                },
                                "textFormat": {
                                    "bold": True
                                }
                            }
                        },
                        "fields": "userEnteredFormat(backgroundColor,textFormat)"
                    }
                }]
                
                spreadsheet_obj.batch_update({"requests": requests})
                logger.info(f"Applied formatting to aggregated row {aggregated_row_num}")
                
            except Exception as e:
                logger.warning(f"Could not format aggregated row: {e}")
            
            kappa_msg = f" with QWK: {qwk:.4f}" if qwk is not None else " (no QWK - no actual scores)"
            logger.info(f"Successfully wrote {len(scores)} scores to Google Sheets starting at row {next_row}{kappa_msg}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing scores to Google Sheets: {e}")
            return False
    
    def export_scores_to_csv(self, 
                            scores: List[Dict[str, Any]], 
                            csv_path: str,
                            run_id: str = None,
                            actual_scores: List[float] = None,
                            component_categories: List = None) -> bool:
        """
        Export scores to a CSV file for ML analysis.
        
        Args:
            scores: List of score dictionaries to export
            csv_path: Path to save the CSV file
            run_id: Run identifier (hour:minute:second format)
            actual_scores: List of actual scores
            component_categories: List of RubricCategory objects for individual columns
        
        Returns:
            True if successful, False otherwise
        """
        if not scores:
            logger.warning("No scores to export to CSV")
            return False
        
        # Generate run_id if not provided
        if not run_id:
            now = datetime.now()
            run_id = now.strftime("%H:%M:%S")
            
        try:
            # Prepare data for CSV export
            csv_data = []
            
            for i, score in enumerate(scores):
                # Get actual score if available
                actual_score = actual_scores[i] if actual_scores and i < len(actual_scores) else None
                
                # Start building row data
                row_data = {
                    "run_id": run_id,
                    "essay_id": score["essay_id"],
                    "ai_score": score["total_score"],
                    "actual_score": actual_score,
                }

                # Compute per-essay means across all categories (including Essay Relevance if present)
                if "category_scores" in score and score["category_scores"]:
                    try:
                        vals = [float(v) for v in score["category_scores"].values() if v is not None]
                        if len(vals) > 0:
                            row_data["mean_components"] = float(np.mean(vals))
                            # Geometric mean and harmonic mean assume positive values; rubric scores are 1..6
                            arr = np.array(vals, dtype=float)
                            row_data["geometric_mean"] = float(np.exp(np.mean(np.log(arr))))
                            row_data["harmonic_mean"] = float(len(arr) / np.sum(1.0 / arr))
                        else:
                            row_data["mean_components"] = None
                            row_data["geometric_mean"] = None
                            row_data["harmonic_mean"] = None
                    except Exception:
                        row_data["mean_components"] = None
                        row_data["geometric_mean"] = None
                        row_data["harmonic_mean"] = None
                else:
                    row_data["mean_components"] = None
                    row_data["geometric_mean"] = None
                    row_data["harmonic_mean"] = None
                
                # Add Essay Relevance score (separately)
                if "category_scores" in score and score["category_scores"]:
                    row_data["essay_relevance"] = score["category_scores"].get("Essay Relevance", None)
                else:
                    row_data["essay_relevance"] = None
                
                # Add individual component scores
                if component_categories:
                    for category in component_categories:
                        # Clean category name for CSV column
                        col_name = category.name.lower().replace(" ", "_").replace("&", "and").replace("(", "").replace(")", "").replace(",", "")
                        category_score = None
                        if "category_scores" in score and score["category_scores"]:
                            category_score = score["category_scores"].get(category.name, None)
                        row_data[col_name] = category_score
                
                csv_data.append(row_data)
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(csv_data)
            
            # Ensure directory exists
            csv_path_obj = Path(csv_path)
            csv_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            logger.info(f"Successfully exported {len(scores)} scores to CSV: {csv_path}")
            
            # Calculate QWK if actual scores are available
            if actual_scores and len(actual_scores) > 0:
                predicted_scores = [score["total_score"] for score in scores]
                if len(actual_scores) == len(predicted_scores):
                    qwk = self.calculate_quadratic_weighted_kappa(actual_scores, predicted_scores)
                    logger.info(f"Calculated QWK for CSV export: {qwk:.4f}")
                    
                    # Optionally add QWK as metadata in a separate file
                    metadata_path = csv_path_obj.with_suffix('.metadata.txt')
                    with open(metadata_path, 'w') as f:
                        f.write(f"Run ID: {run_id}\n")
                        f.write(f"Number of essays: {len(scores)}\n")
                        f.write(f"QWK: {qwk:.4f}\n")
                        f.write(f"Export timestamp: {datetime.now().isoformat()}\n")
                    logger.info(f"Saved metadata to: {metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting scores to CSV: {e}")
            return False