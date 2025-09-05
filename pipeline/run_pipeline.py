#!/usr/bin/env python3
"""
Main Pipeline Orchestrator
Coordinates the complete AI paper processing pipeline
"""

import argparse
import json
import logging
import os
import sys
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add pipeline directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fetcher import ArxivPaperFetcher
from processor import ClaudeProcessor
from generator import HTMLGenerator

class PipelineOrchestrator:
    """Main orchestrator for the AI paper processing pipeline"""
    
    def __init__(self, config_path: str = None):
        """Initialize the pipeline with configuration"""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_directories()
        
        # Initialize components
        self.fetcher = ArxivPaperFetcher(self._get_fetcher_config())
        self.processor = ClaudeProcessor(self._get_processor_config())
        self.generator = HTMLGenerator(self._get_generator_config())
        
        self.logger.info("Pipeline initialized successfully")
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent / 'config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Expand environment variables
        self._expand_env_vars(config)
        
        return config
    
    def _expand_env_vars(self, config: Dict):
        """Recursively expand environment variables in config"""
        for key, value in config.items():
            if isinstance(value, dict):
                self._expand_env_vars(value)
            elif isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                config[key] = os.environ.get(env_var, '')
    
    def _setup_logging(self):
        """Set up logging configuration"""
        log_config = self.config.get('logging', {})
        
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_config.get('file', 'pipeline.log'))
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_directories(self):
        """Create necessary directories"""
        dirs = self.config.get('directories', {})
        for dir_name, dir_path in dirs.items():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _get_fetcher_config(self) -> Dict:
        """Get configuration for fetcher"""
        arxiv_config = self.config.get('arxiv', {})
        return {
            'categories': arxiv_config.get('categories', []),
            'keywords': arxiv_config.get('keywords', []),
            'max_results_per_category': arxiv_config.get('max_papers_per_category', 100),
            'min_relevance_score': arxiv_config.get('min_relevance_score', 1.0),
            'rate_limit_seconds': arxiv_config.get('rate_limit_seconds', 3),
            'data_dir': self.config['directories']['data']
        }
    
    def _get_processor_config(self) -> Dict:
        """Get configuration for processor"""
        claude_config = self.config.get('claude', {})
        return {
            'api_key': claude_config.get('api_key', ''),
            'model': claude_config.get('model', 'claude-3-opus-20240229'),
            'max_tokens': claude_config.get('max_tokens', 4000),
            'temperature': claude_config.get('temperature', 0.3),
            'batch_size': claude_config.get('batch_size', 5),
            'rate_limit_delay': claude_config.get('rate_limit_delay', 2),
            'data_dir': self.config['directories']['data'],
            'template_path': Path(self.config['directories']['templates']) / 'summary-template.md'
        }
    
    def _get_generator_config(self) -> Dict:
        """Get configuration for HTML generator"""
        gen_config = self.config.get('generator', {})
        return {
            'output_dir': self.config['directories']['output'],
            'site_title': gen_config.get('site_title', 'AICOE Research Library'),
            'site_description': gen_config.get('site_description', 'Daily AI Papers'),
            'base_url': gen_config.get('base_url', ''),
            'papers_per_page': gen_config.get('papers_per_page', 20),
            'generate_rss': gen_config.get('generate_rss', True)
        }
    
    def run_pipeline(self, date: datetime = None, force: bool = False):
        """
        Run the complete pipeline
        
        Args:
            date: Date to fetch papers for (default: yesterday)
            force: Force refetch even if papers were already processed
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting AI Paper Processing Pipeline")
        self.logger.info("=" * 60)
        
        try:
            # Step 1: Fetch papers from arXiv
            self.logger.info("Step 1: Fetching papers from arXiv")
            papers = self.fetcher.fetch_papers_by_date(date, force_refetch=force)
            
            if not papers:
                self.logger.info("No new papers found. Pipeline complete.")
                return
            
            self.logger.info(f"Fetched {len(papers)} papers")
            
            # Save raw papers
            raw_file = self.fetcher.save_papers(papers)
            
            # Step 2: Process papers with Claude AI
            if self.config['features'].get('ai_processing', True):
                self.logger.info("Step 2: Processing papers with Claude AI")
                
                # Limit papers per run
                max_papers = self.config['processing'].get('max_papers_per_run', 50)
                papers_to_process = papers[:max_papers]
                
                processed_papers = self.processor.process_papers_batch(papers_to_process)
                self.logger.info(f"Processed {len(processed_papers)} papers")
            else:
                self.logger.info("Step 2: Skipping AI processing (disabled)")
                # Create basic summaries without AI
                processed_papers = self._create_basic_summaries(papers)
            
            # Step 3: Generate HTML website
            self.logger.info("Step 3: Generating HTML website")
            self.generator.generate_site(processed_papers)
            self.logger.info("Website generated successfully")
            
            # Step 4: Archive old papers if needed
            if self.config['features'].get('auto_archive', True):
                self._archive_old_papers()
            
            # Step 5: Generate statistics
            if self.config['processing'].get('generate_stats', True):
                self._generate_statistics(papers, processed_papers)
            
            self.logger.info("=" * 60)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info("=" * 60)
            
            # Print summary
            self._print_summary(papers, processed_papers)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    def _create_basic_summaries(self, papers: List[Dict]) -> List[Dict]:
        """Create basic summaries without AI processing"""
        processed = []
        for paper in papers:
            processed.append({
                'paper_id': paper['id'],
                'title': paper['title'],
                'authors': paper['authors'],
                'categories': paper['categories'],
                'arxiv_url': paper['arxiv_url'],
                'pdf_url': paper['pdf_url'],
                'published_date': paper['published'],
                'relevance_score': paper.get('relevance_score', 0),
                'summary_markdown': f"## Abstract\n\n{paper['summary']}",
                'processed_date': datetime.now().isoformat()
            })
        return processed
    
    def _archive_old_papers(self):
        """Archive papers older than retention period"""
        retention_days = self.config['processing'].get('retention_days', 30)
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        archive_dir = Path(self.config['directories']['archive'])
        processed_dir = Path(self.config['directories']['processed'])
        
        archived_count = 0
        for file in processed_dir.glob('*.json'):
            # Check file modification time
            mtime = datetime.fromtimestamp(file.stat().st_mtime)
            if mtime < cutoff_date:
                # Move to archive
                archive_file = archive_dir / file.name
                file.rename(archive_file)
                archived_count += 1
        
        if archived_count > 0:
            self.logger.info(f"Archived {archived_count} old papers")
    
    def _generate_statistics(self, fetched: List[Dict], processed: List[Dict]):
        """Generate pipeline statistics"""
        stats = {
            'run_date': datetime.now().isoformat(),
            'papers_fetched': len(fetched),
            'papers_processed': len(processed),
            'categories': {},
            'relevance_distribution': {
                'high': len([p for p in fetched if p.get('relevance_score', 0) > 5]),
                'medium': len([p for p in fetched if 2 < p.get('relevance_score', 0) <= 5]),
                'low': len([p for p in fetched if p.get('relevance_score', 0) <= 2])
            }
        }
        
        # Count by category
        for paper in fetched:
            for cat in paper['categories']:
                stats['categories'][cat] = stats['categories'].get(cat, 0) + 1
        
        # Save statistics
        stats_file = Path(self.config['directories']['data']) / f"stats_{datetime.now().strftime('%Y%m%d')}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Statistics saved to {stats_file}")
    
    def _print_summary(self, fetched: List[Dict], processed: List[Dict]):
        """Print pipeline execution summary"""
        print("\n" + "=" * 60)
        print("📊 PIPELINE SUMMARY")
        print("=" * 60)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"📥 Papers fetched: {len(fetched)}")
        print(f"🤖 Papers processed: {len(processed)}")
        
        if fetched:
            print(f"\n🏆 Top 5 papers by relevance:")
            for i, paper in enumerate(fetched[:5], 1):
                score = paper.get('relevance_score', 0)
                title = paper['title'][:60] + ('...' if len(paper['title']) > 60 else '')
                print(f"  {i}. [{score:.1f}] {title}")
        
        print(f"\n📁 Output location: {self.config['directories']['output']}")
        print(f"🌐 Open {self.config['directories']['output']}/index.html to view")
        print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run AI Paper Processing Pipeline')
    parser.add_argument('--date', type=str, help='Date to fetch papers (YYYY-MM-DD)')
    parser.add_argument('--days-back', type=int, help='Fetch papers from N days ago')
    parser.add_argument('--force', action='store_true', help='Force refetch')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--dry-run', action='store_true', help='Dry run without processing')
    
    args = parser.parse_args()
    
    # Determine date
    if args.date:
        date = datetime.strptime(args.date, '%Y-%m-%d')
    elif args.days_back:
        date = datetime.now() - timedelta(days=args.days_back)
    else:
        date = None  # Will default to yesterday
    
    # Initialize and run pipeline
    orchestrator = PipelineOrchestrator(args.config)
    
    if args.dry_run:
        print("DRY RUN MODE - No actual processing will occur")
        orchestrator.logger.info("Running in dry-run mode")
    else:
        orchestrator.run_pipeline(date, force=args.force)


if __name__ == '__main__':
    main()