"""Report Scheduler for Automated Report Generation.

This module provides scheduling capabilities for automated report generation,
including recurring reports, email delivery, and report management.
"""

import json
import logging
import smtplib
import threading
import time
from datetime import datetime, timedelta
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

import schedule

from .report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class ScheduledReport:
    """Represents a scheduled report configuration."""
    
    def __init__(self, report_id: str, name: str, template_id: str,
                 schedule_config: Dict[str, Any], parameters: Dict[str, Any],
                 delivery_config: Dict[str, Any]):
        """Initialize a scheduled report.
        
        Args:
            report_id: Unique report identifier
            name: Human-readable report name
            template_id: Report template to use
            schedule_config: Scheduling configuration
            parameters: Report parameters
            delivery_config: Delivery configuration (email, file, etc.)
        """
        self.report_id = report_id
        self.name = name
        self.template_id = template_id
        self.schedule_config = schedule_config
        self.parameters = parameters
        self.delivery_config = delivery_config
        self.created_at = datetime.utcnow()
        self.last_run = None
        self.next_run = None
        self.enabled = True
        self.run_count = 0
        self.error_count = 0
        self.last_error = None


class ReportScheduler:
    """Advanced report scheduling and delivery system."""
    
    def __init__(self, report_generator: ReportGenerator):
        """Initialize the report scheduler.
        
        Args:
            report_generator: Report generator instance
        """
        self.report_generator = report_generator
        self.scheduled_reports = {}
        self.scheduler_thread = None
        self.running = False
        
        # Setup directories
        self.schedules_dir = Path("analytics/schedules")
        self.schedules_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing scheduled reports
        self._load_scheduled_reports()
        
        self.logger = logging.getLogger(__name__)
    
    def create_scheduled_report(self, name: str, template_id: str,
                              schedule_config: Dict[str, Any],
                              parameters: Dict[str, Any] = None,
                              delivery_config: Dict[str, Any] = None) -> str:
        """Create a new scheduled report.
        
        Args:
            name: Report name
            template_id: Template to use
            schedule_config: Schedule configuration
            parameters: Report parameters
            delivery_config: Delivery configuration
            
        Returns:
            Scheduled report ID
        """
        # Generate unique report ID
        report_id = f"scheduled_{int(datetime.utcnow().timestamp())}"
        
        # Validate schedule configuration
        self._validate_schedule_config(schedule_config)
        
        # Create scheduled report
        scheduled_report = ScheduledReport(
            report_id=report_id,
            name=name,
            template_id=template_id,
            schedule_config=schedule_config,
            parameters=parameters or {},
            delivery_config=delivery_config or {}
        )
        
        # Calculate next run time
        scheduled_report.next_run = self._calculate_next_run(schedule_config)
        
        # Store the scheduled report
        self.scheduled_reports[report_id] = scheduled_report
        
        # Save to disk
        self._save_scheduled_report(scheduled_report)
        
        # Update scheduler if running
        if self.running:
            self._update_scheduler()
        
        self.logger.info(f"Created scheduled report: {name} ({report_id})")
        return report_id
    
    def update_scheduled_report(self, report_id: str, **updates) -> bool:
        """Update an existing scheduled report.
        
        Args:
            report_id: Report ID to update
            **updates: Fields to update
            
        Returns:
            True if successful, False otherwise
        """
        if report_id not in self.scheduled_reports:
            self.logger.error(f"Scheduled report not found: {report_id}")
            return False
        
        scheduled_report = self.scheduled_reports[report_id]
        
        # Update fields
        for field, value in updates.items():
            if hasattr(scheduled_report, field):
                setattr(scheduled_report, field, value)
        
        # Recalculate next run if schedule changed
        if 'schedule_config' in updates:
            scheduled_report.next_run = self._calculate_next_run(scheduled_report.schedule_config)
        
        # Save changes
        self._save_scheduled_report(scheduled_report)
        
        # Update scheduler if running
        if self.running:
            self._update_scheduler()
        
        self.logger.info(f"Updated scheduled report: {report_id}")
        return True
    
    def delete_scheduled_report(self, report_id: str) -> bool:
        """Delete a scheduled report.
        
        Args:
            report_id: Report ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        if report_id not in self.scheduled_reports:
            self.logger.error(f"Scheduled report not found: {report_id}")
            return False
        
        # Remove from memory
        del self.scheduled_reports[report_id]
        
        # Remove from disk
        schedule_file = self.schedules_dir / f"{report_id}.json"
        if schedule_file.exists():
            schedule_file.unlink()
        
        # Update scheduler if running
        if self.running:
            self._update_scheduler()
        
        self.logger.info(f"Deleted scheduled report: {report_id}")
        return True
    
    def start_scheduler(self):
        """Start the report scheduler."""
        if self.running:
            self.logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self._update_scheduler()
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("Report scheduler started")
    
    def stop_scheduler(self):
        """Stop the report scheduler."""
        self.running = False
        schedule.clear()
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        self.logger.info("Report scheduler stopped")
    
    def run_scheduled_report(self, report_id: str, force: bool = False) -> Dict[str, Any]:
        """Run a scheduled report immediately.
        
        Args:
            report_id: Report ID to run
            force: Force run even if disabled
            
        Returns:
            Execution results
        """
        if report_id not in self.scheduled_reports:
            return {'error': f'Scheduled report not found: {report_id}'}
        
        scheduled_report = self.scheduled_reports[report_id]
        
        if not scheduled_report.enabled and not force:
            return {'error': 'Scheduled report is disabled'}
        
        try:
            self.logger.info(f"Running scheduled report: {scheduled_report.name}")
            
            # Generate the report
            result = self.report_generator.generate_report(
                template_id=scheduled_report.template_id,
                parameters=scheduled_report.parameters,
                output_format=scheduled_report.delivery_config.get('format', 'pdf')
            )
            
            if 'error' in result:
                scheduled_report.error_count += 1
                scheduled_report.last_error = result['error']
                self.logger.error(f"Report generation failed: {result['error']}")
                return result
            
            # Deliver the report
            delivery_result = self._deliver_report(scheduled_report, result)
            
            # Update execution stats
            scheduled_report.last_run = datetime.utcnow()
            scheduled_report.run_count += 1
            scheduled_report.next_run = self._calculate_next_run(scheduled_report.schedule_config)
            
            # Save updated schedule
            self._save_scheduled_report(scheduled_report)
            
            # Combine results
            result.update(delivery_result)
            
            self.logger.info(f"Successfully executed scheduled report: {scheduled_report.name}")
            return result
            
        except Exception as e:
            scheduled_report.error_count += 1
            scheduled_report.last_error = str(e)
            self.logger.error(f"Failed to execute scheduled report {report_id}: {e}")
            return {'error': str(e)}
    
    def get_scheduled_reports(self) -> List[Dict[str, Any]]:
        """Get list of all scheduled reports."""
        reports = []
        
        for report_id, scheduled_report in self.scheduled_reports.items():
            reports.append({
                'report_id': report_id,
                'name': scheduled_report.name,
                'template_id': scheduled_report.template_id,
                'schedule_config': scheduled_report.schedule_config,
                'delivery_config': scheduled_report.delivery_config,
                'enabled': scheduled_report.enabled,
                'created_at': scheduled_report.created_at.isoformat(),
                'last_run': scheduled_report.last_run.isoformat() if scheduled_report.last_run else None,
                'next_run': scheduled_report.next_run.isoformat() if scheduled_report.next_run else None,
                'run_count': scheduled_report.run_count,
                'error_count': scheduled_report.error_count,
                'last_error': scheduled_report.last_error
            })
        
        return reports
    
    def get_scheduled_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific scheduled report."""
        if report_id not in self.scheduled_reports:
            return None
        
        scheduled_report = self.scheduled_reports[report_id]
        
        return {
            'report_id': report_id,
            'name': scheduled_report.name,
            'template_id': scheduled_report.template_id,
            'schedule_config': scheduled_report.schedule_config,
            'parameters': scheduled_report.parameters,
            'delivery_config': scheduled_report.delivery_config,
            'enabled': scheduled_report.enabled,
            'created_at': scheduled_report.created_at.isoformat(),
            'last_run': scheduled_report.last_run.isoformat() if scheduled_report.last_run else None,
            'next_run': scheduled_report.next_run.isoformat() if scheduled_report.next_run else None,
            'run_count': scheduled_report.run_count,
            'error_count': scheduled_report.error_count,
            'last_error': scheduled_report.last_error
        }
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                time.sleep(60)
    
    def _update_scheduler(self):
        """Update the scheduler with current scheduled reports."""
        schedule.clear()
        
        for report_id, scheduled_report in self.scheduled_reports.items():
            if not scheduled_report.enabled:
                continue
            
            schedule_config = scheduled_report.schedule_config
            schedule_type = schedule_config.get('type', 'daily')
            
            if schedule_type == 'daily':
                time_str = schedule_config.get('time', '09:00')
                schedule.every().day.at(time_str).do(self._execute_scheduled_report, report_id)
            
            elif schedule_type == 'weekly':
                day_of_week = schedule_config.get('day_of_week', 'monday')
                time_str = schedule_config.get('time', '09:00')
                getattr(schedule.every(), day_of_week).at(time_str).do(self._execute_scheduled_report, report_id)
            
            elif schedule_type == 'monthly':
                # For monthly, we'll check daily and run on the specified day
                time_str = schedule_config.get('time', '09:00')
                schedule.every().day.at(time_str).do(self._check_monthly_report, report_id)
            
            elif schedule_type == 'hourly':
                schedule.every().hour.do(self._execute_scheduled_report, report_id)
            
            elif schedule_type == 'interval':
                interval_hours = schedule_config.get('interval_hours', 24)
                schedule.every(interval_hours).hours.do(self._execute_scheduled_report, report_id)
    
    def _execute_scheduled_report(self, report_id: str):
        """Execute a scheduled report (called by scheduler)."""
        try:
            result = self.run_scheduled_report(report_id)
            if 'error' in result:
                self.logger.error(f"Scheduled report execution failed: {result['error']}")
        except Exception as e:
            self.logger.error(f"Exception during scheduled report execution: {e}")
    
    def _check_monthly_report(self, report_id: str):
        """Check if monthly report should run today."""
        if report_id not in self.scheduled_reports:
            return
        
        scheduled_report = self.scheduled_reports[report_id]
        schedule_config = scheduled_report.schedule_config
        
        if schedule_config.get('type') != 'monthly':
            return
        
        today = datetime.now()
        day_of_month = schedule_config.get('day_of_month', 1)
        
        # Run on the specified day of month, or last day if month is shorter
        if today.day == day_of_month or (day_of_month > 28 and today.day == self._last_day_of_month(today)):
            self._execute_scheduled_report(report_id)
    
    def _last_day_of_month(self, date: datetime) -> int:
        """Get the last day of the month for a given date."""
        if date.month == 12:
            next_month = date.replace(year=date.year + 1, month=1, day=1)
        else:
            next_month = date.replace(month=date.month + 1, day=1)
        
        last_day = next_month - timedelta(days=1)
        return last_day.day
    
    def _calculate_next_run(self, schedule_config: Dict[str, Any]) -> datetime:
        """Calculate the next run time for a schedule."""
        now = datetime.now()
        schedule_type = schedule_config.get('type', 'daily')
        
        if schedule_type == 'daily':
            time_str = schedule_config.get('time', '09:00')
            hour, minute = map(int, time_str.split(':'))
            
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            
            return next_run
        
        elif schedule_type == 'weekly':
            day_of_week = schedule_config.get('day_of_week', 'monday')
            time_str = schedule_config.get('time', '09:00')
            hour, minute = map(int, time_str.split(':'))
            
            # Map day names to numbers (0 = Monday)
            day_map = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            
            target_day = day_map.get(day_of_week.lower(), 0)
            days_ahead = target_day - now.weekday()
            
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            
            next_run = now + timedelta(days=days_ahead)
            next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            return next_run
        
        elif schedule_type == 'monthly':
            day_of_month = schedule_config.get('day_of_month', 1)
            time_str = schedule_config.get('time', '09:00')
            hour, minute = map(int, time_str.split(':'))
            
            next_run = now.replace(day=day_of_month, hour=hour, minute=minute, second=0, microsecond=0)
            
            if next_run <= now:
                # Move to next month
                if now.month == 12:
                    next_run = next_run.replace(year=now.year + 1, month=1)
                else:
                    next_run = next_run.replace(month=now.month + 1)
            
            return next_run
        
        elif schedule_type == 'hourly':
            return now + timedelta(hours=1)
        
        elif schedule_type == 'interval':
            interval_hours = schedule_config.get('interval_hours', 24)
            return now + timedelta(hours=interval_hours)
        
        else:
            # Default to daily
            return now + timedelta(days=1)
    
    def _deliver_report(self, scheduled_report: ScheduledReport, report_result: Dict[str, Any]) -> Dict[str, Any]:
        """Deliver a generated report according to delivery configuration."""
        delivery_config = scheduled_report.delivery_config
        delivery_methods = delivery_config.get('methods', ['file'])
        
        delivery_results = {}
        
        for method in delivery_methods:
            try:
                if method == 'email':
                    result = self._deliver_via_email(scheduled_report, report_result, delivery_config)
                elif method == 'file':
                    result = self._deliver_via_file(scheduled_report, report_result, delivery_config)
                elif method == 'webhook':
                    result = self._deliver_via_webhook(scheduled_report, report_result, delivery_config)
                else:
                    result = {'error': f'Unknown delivery method: {method}'}
                
                delivery_results[method] = result
                
            except Exception as e:
                delivery_results[method] = {'error': str(e)}
                self.logger.error(f"Delivery failed for method {method}: {e}")
        
        return {'delivery_results': delivery_results}
    
    def _deliver_via_email(self, scheduled_report: ScheduledReport, 
                          report_result: Dict[str, Any], delivery_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deliver report via email."""
        email_config = delivery_config.get('email', {})
        
        # Email configuration
        smtp_server = email_config.get('smtp_server', 'localhost')
        smtp_port = email_config.get('smtp_port', 587)
        username = email_config.get('username')
        password = email_config.get('password')
        use_tls = email_config.get('use_tls', True)
        
        # Email content
        recipients = email_config.get('recipients', [])
        subject = email_config.get('subject', f"Scheduled Report: {scheduled_report.name}")
        body = email_config.get('body', f"""
Automated report generated by Mimir Analytics.

Report: {scheduled_report.name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Template: {scheduled_report.template_id}

Please find the report attached.
        """)
        
        if not recipients:
            return {'error': 'No recipients specified'}
        
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = email_config.get('from_address', username)
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach report file
            if 'file_path' in report_result:
                file_path = Path(report_result['file_path'])
                
                with open(file_path, 'rb') as f:
                    attachment = MIMEApplication(f.read())
                    attachment.add_header(
                        'Content-Disposition',
                        'attachment',
                        filename=file_path.name
                    )
                    msg.attach(attachment)
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if use_tls:
                    server.starttls()
                
                if username and password:
                    server.login(username, password)
                
                server.send_message(msg)
            
            return {
                'success': True,
                'recipients': recipients,
                'subject': subject
            }
            
        except Exception as e:
            return {'error': f'Email delivery failed: {str(e)}'}
    
    def _deliver_via_file(self, scheduled_report: ScheduledReport,
                         report_result: Dict[str, Any], delivery_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deliver report by saving to specified location."""
        file_config = delivery_config.get('file', {})
        
        if 'file_path' not in report_result:
            return {'error': 'No file generated in report result'}
        
        source_path = Path(report_result['file_path'])
        
        # Determine destination
        destination_dir = file_config.get('destination_dir', 'analytics/scheduled_reports')
        destination_path = Path(destination_dir)
        destination_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = file_config.get('filename_pattern', '{report_name}_{timestamp}.{ext}')
        filename = filename.format(
            report_name=scheduled_report.name.replace(' ', '_'),
            timestamp=timestamp,
            ext=source_path.suffix[1:]  # Remove the dot
        )
        
        final_path = destination_path / filename
        
        try:
            # Copy file to destination
            import shutil
            shutil.copy2(source_path, final_path)
            
            return {
                'success': True,
                'file_path': str(final_path),
                'file_size': final_path.stat().st_size
            }
            
        except Exception as e:
            return {'error': f'File delivery failed: {str(e)}'}
    
    def _deliver_via_webhook(self, scheduled_report: ScheduledReport,
                           report_result: Dict[str, Any], delivery_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deliver report via webhook."""
        webhook_config = delivery_config.get('webhook', {})
        
        url = webhook_config.get('url')
        if not url:
            return {'error': 'No webhook URL specified'}
        
        # Prepare payload
        payload = {
            'report_id': scheduled_report.report_id,
            'report_name': scheduled_report.name,
            'template_id': scheduled_report.template_id,
            'generated_at': datetime.now().isoformat(),
            'file_path': report_result.get('file_path'),
            'file_size': report_result.get('file_size'),
            'format': report_result.get('format')
        }
        
        # Add custom fields
        custom_fields = webhook_config.get('custom_fields', {})
        payload.update(custom_fields)
        
        try:
            import requests
            
            headers = webhook_config.get('headers', {'Content-Type': 'application/json'})
            timeout = webhook_config.get('timeout', 30)
            
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            return {
                'success': True,
                'status_code': response.status_code,
                'response': response.text[:500]  # Truncate response
            }
            
        except Exception as e:
            return {'error': f'Webhook delivery failed: {str(e)}'}
    
    def _validate_schedule_config(self, schedule_config: Dict[str, Any]):
        """Validate schedule configuration."""
        schedule_type = schedule_config.get('type', 'daily')
        
        valid_types = ['daily', 'weekly', 'monthly', 'hourly', 'interval']
        if schedule_type not in valid_types:
            raise ValueError(f"Invalid schedule type: {schedule_type}")
        
        if schedule_type in ['daily', 'weekly', 'monthly']:
            time_str = schedule_config.get('time', '09:00')
            try:
                hour, minute = map(int, time_str.split(':'))
                if not (0 <= hour <= 23) or not (0 <= minute <= 59):
                    raise ValueError("Invalid time format")
            except ValueError:
                raise ValueError(f"Invalid time format: {time_str}")
        
        if schedule_type == 'weekly':
            day_of_week = schedule_config.get('day_of_week', 'monday').lower()
            valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            if day_of_week not in valid_days:
                raise ValueError(f"Invalid day of week: {day_of_week}")
        
        if schedule_type == 'monthly':
            day_of_month = schedule_config.get('day_of_month', 1)
            if not (1 <= day_of_month <= 31):
                raise ValueError(f"Invalid day of month: {day_of_month}")
        
        if schedule_type == 'interval':
            interval_hours = schedule_config.get('interval_hours', 24)
            if not isinstance(interval_hours, int) or interval_hours < 1:
                raise ValueError(f"Invalid interval hours: {interval_hours}")
    
    def _load_scheduled_reports(self):
        """Load scheduled reports from disk."""
        for schedule_file in self.schedules_dir.glob("*.json"):
            try:
                with open(schedule_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                scheduled_report = ScheduledReport(
                    report_id=data['report_id'],
                    name=data['name'],
                    template_id=data['template_id'],
                    schedule_config=data['schedule_config'],
                    parameters=data['parameters'],
                    delivery_config=data['delivery_config']
                )
                
                # Restore state
                scheduled_report.created_at = datetime.fromisoformat(data['created_at'])
                scheduled_report.enabled = data.get('enabled', True)
                scheduled_report.run_count = data.get('run_count', 0)
                scheduled_report.error_count = data.get('error_count', 0)
                
                if data.get('last_run'):
                    scheduled_report.last_run = datetime.fromisoformat(data['last_run'])
                
                # Calculate next run
                scheduled_report.next_run = self._calculate_next_run(scheduled_report.schedule_config)
                
                self.scheduled_reports[scheduled_report.report_id] = scheduled_report
                
                self.logger.info(f"Loaded scheduled report: {scheduled_report.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load scheduled report from {schedule_file}: {e}")
    
    def _save_scheduled_report(self, scheduled_report: ScheduledReport):
        """Save a scheduled report to disk."""
        schedule_file = self.schedules_dir / f"{scheduled_report.report_id}.json"
        
        data = {
            'report_id': scheduled_report.report_id,
            'name': scheduled_report.name,
            'template_id': scheduled_report.template_id,
            'schedule_config': scheduled_report.schedule_config,
            'parameters': scheduled_report.parameters,
            'delivery_config': scheduled_report.delivery_config,
            'created_at': scheduled_report.created_at.isoformat(),
            'enabled': scheduled_report.enabled,
            'run_count': scheduled_report.run_count,
            'error_count': scheduled_report.error_count,
            'last_run': scheduled_report.last_run.isoformat() if scheduled_report.last_run else None
        }
        
        try:
            with open(schedule_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save scheduled report {scheduled_report.report_id}: {e}")