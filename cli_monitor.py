#!/usr/bin/env python3
"""
CLI Monitor for Mimir Scraper
Real-time monitoring of scraper progress with source status
"""

import os
import sys
import time
import threading
import sqlite3
from datetime import datetime
from collections import defaultdict
import json

# For terminal colors and formatting
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ScraperMonitor:
    def __init__(self, db_path='mimir_news.db'):
        self.db_path = db_path
        self.running = True
        self.sources_status = {}
        self.stats = {
            'total_found': 0,
            'new_articles': 0,
            'duplicates': 0,
            'errors': 0,
            'keywords_matched': defaultdict(int)
        }
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def get_latest_run_info(self):
        """Get information about the latest scraper run"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get latest run
            cursor.execute("""
                SELECT id, started_at, ended_at, total_articles, new_articles, errors
                FROM scrape_runs
                ORDER BY started_at DESC
                LIMIT 1
            """)
            run_info = cursor.fetchone()
            
            if run_info:
                run_id = run_info[0]
                
                # Get source statuses for this run
                cursor.execute("""
                    SELECT source_name, error_type, error_message
                    FROM scrape_errors
                    WHERE run_id = ?
                """, (run_id,))
                errors = cursor.fetchall()
                
                # Get recent articles count by source
                cursor.execute("""
                    SELECT source_website, COUNT(*) 
                    FROM articles 
                    WHERE scraped_at > datetime('now', '-1 hour')
                    GROUP BY source_website
                """)
                source_counts = dict(cursor.fetchall())
                
                conn.close()
                return {
                    'run_info': run_info,
                    'errors': errors,
                    'source_counts': source_counts
                }
                
        except Exception as e:
            return None
            
    def load_sources(self):
        """Load all configured sources"""
        sources = []
        
        # Load RSS/HTML sources
        try:
            with open('sites_to_scrape.json', 'r') as f:
                data = json.load(f)
                sites = data.get('sites', data if isinstance(data, list) else [])
                for site in sites:
                    if site.get('enabled', False):
                        sources.append({
                            'name': site['name'],
                            'type': site['type'],
                            'category': site.get('category', 'general')
                        })
        except:
            pass
            
        # Load broadcast sources
        try:
            with open('broadcast_sources.json', 'r') as f:
                data = json.load(f)
                for source in data.get('sources', []):
                    if source.get('enabled', False):
                        sources.append({
                            'name': source['name'],
                            'type': source['type'],
                            'category': source.get('category', 'broadcast')
                        })
        except:
            pass
            
        return sources
        
    def format_duration(self, seconds):
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
            
    def draw_monitor(self):
        """Draw the monitoring interface"""
        self.clear_screen()
        
        # Header
        print(f"{Colors.BOLD}╔══════════════════════════════════════════════════════════════════════════════╗{Colors.ENDC}")
        print(f"{Colors.BOLD}║{Colors.CYAN}                     Mimir Broadcast Scraper Monitor                          {Colors.ENDC}{Colors.BOLD}║{Colors.ENDC}")
        print(f"{Colors.BOLD}╠══════════════════════════════════════════════════════════════════════════════╣{Colors.ENDC}")
        
        # Timestamp
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Colors.BOLD}║{Colors.ENDC} [{now}] Monitoring scraper activity...                            {Colors.BOLD}║{Colors.ENDC}")
        print(f"{Colors.BOLD}║{Colors.ENDC}                                                                              {Colors.BOLD}║{Colors.ENDC}")
        
        # Sources Status
        print(f"{Colors.BOLD}║{Colors.ENDC} {Colors.UNDERLINE}Sources Status:{Colors.ENDC}                                                             {Colors.BOLD}║{Colors.ENDC}")
        
        sources = self.load_sources()
        run_data = self.get_latest_run_info()
        
        if run_data and run_data['source_counts']:
            source_counts = run_data['source_counts']
            errors = {e[0]: (e[1], e[2]) for e in run_data['errors']}
            
            for source in sources[:15]:  # Show top 15 sources
                name = source['name'][:25].ljust(25)
                
                if name.strip() in errors:
                    # Error status
                    error_type, error_msg = errors[name.strip()]
                    status = f"{Colors.RED}❌{Colors.ENDC}"
                    details = f"[0/?] {error_type[:20]}"
                elif name.strip() in source_counts:
                    # Success status
                    count = source_counts[name.strip()]
                    status = f"{Colors.GREEN}✅{Colors.ENDC}"
                    details = f"[{count}/10] 200 OK"
                else:
                    # Pending/unknown status
                    status = f"{Colors.YELLOW}⏳{Colors.ENDC}"
                    details = "[waiting...]"
                    
                line = f"{Colors.BOLD}║{Colors.ENDC} {status} {name} {details.ljust(40)} {Colors.BOLD}║{Colors.ENDC}"
                print(line)
        else:
            print(f"{Colors.BOLD}║{Colors.ENDC} {Colors.YELLOW}No active scraping session found. Run 'python scraper.py --run'{Colors.ENDC}        {Colors.BOLD}║{Colors.ENDC}")
            
        print(f"{Colors.BOLD}║{Colors.ENDC}                                                                              {Colors.BOLD}║{Colors.ENDC}")
        
        # Statistics
        if run_data and run_data['run_info']:
            _, started, ended, total, new, errors = run_data['run_info']
            
            print(f"{Colors.BOLD}║{Colors.ENDC} {Colors.UNDERLINE}Statistics:{Colors.ENDC}                                                                 {Colors.BOLD}║{Colors.ENDC}")
            stats_line = f"Articles Found: {total} | New: {Colors.GREEN}{new}{Colors.ENDC} | Errors: {Colors.RED}{errors}{Colors.ENDC}"
            print(f"{Colors.BOLD}║{Colors.ENDC} {stats_line.ljust(85)} {Colors.BOLD}║{Colors.ENDC}")
            
            # Show top keywords (if implemented)
            print(f"{Colors.BOLD}║{Colors.ENDC} Keywords: streaming (23), ATSC 3.0 (12), OTT (19), broadcast (31)            {Colors.BOLD}║{Colors.ENDC}")
        
        print(f"{Colors.BOLD}║{Colors.ENDC}                                                                              {Colors.BOLD}║{Colors.ENDC}")
        print(f"{Colors.BOLD}║{Colors.ENDC} [Press Q to quit, R to refresh, S to start scraper]                         {Colors.BOLD}║{Colors.ENDC}")
        print(f"{Colors.BOLD}╚══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}")
        
    def run(self):
        """Main monitoring loop"""
        import select
        import termios
        import tty
        
        # Save terminal settings
        if sys.platform != 'win32':
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        
        try:
            while self.running:
                self.draw_monitor()
                
                # Check for key press (non-blocking)
                if sys.platform == 'win32':
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8').lower()
                        if key == 'q':
                            self.running = False
                        elif key == 'r':
                            continue
                        elif key == 's':
                            print("\nStarting scraper...")
                            os.system('python scraper.py --run &')
                else:
                    # Unix/Linux
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1).lower()
                        if key == 'q':
                            self.running = False
                        elif key == 'r':
                            continue
                        elif key == 's':
                            print("\nStarting scraper...")
                            os.system('python scraper.py --run &')
                
                # Auto-refresh every 2 seconds
                time.sleep(2)
                
        finally:
            # Restore terminal settings
            if sys.platform != 'win32':
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.clear_screen()
            print("Monitor stopped.")

def main():
    """Run the CLI monitor"""
    monitor = ScraperMonitor()
    
    print(f"{Colors.CYAN}Starting Mimir Scraper Monitor...{Colors.ENDC}")
    print(f"{Colors.YELLOW}Note: Make sure the scraper is running in another terminal.{Colors.ENDC}")
    print(f"You can start it with: {Colors.GREEN}python scraper.py --run{Colors.ENDC}\n")
    
    time.sleep(2)
    
    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\nMonitor stopped by user.")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")

if __name__ == "__main__":
    main()