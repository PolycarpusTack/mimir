"""Data Export System for Mimir Analytics.

This module provides comprehensive data export capabilities including multiple
formats, bulk exports, streaming for large datasets, and scheduled exports.
"""

import csv
import gzip
import io
import json
import logging
import os
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xlsxwriter

try:
    import avro
    import avro.io
    import avro.datafile
    import avro.schema
    AVRO_AVAILABLE = True
except ImportError:
    AVRO_AVAILABLE = False

from .data_warehouse import AnalyticsDataWarehouse
from db_adapter import DatabaseAdapter

logger = logging.getLogger(__name__)


class ExportFormat:
    """Supported export formats."""
    CSV = 'csv'
    JSON = 'json'
    JSONL = 'jsonl'  # JSON Lines format
    EXCEL = 'excel'
    PARQUET = 'parquet'
    AVRO = 'avro'
    TSV = 'tsv'
    XML = 'xml'
    SQL = 'sql'


class ExportConfig:
    """Export configuration settings."""
    
    def __init__(self, format: str, compression: bool = False,
                 include_metadata: bool = True, chunk_size: int = 10000,
                 filters: Dict[str, Any] = None, columns: List[str] = None):
        """Initialize export configuration.
        
        Args:
            format: Export format (from ExportFormat)
            compression: Whether to compress the output
            include_metadata: Include metadata in export
            chunk_size: Number of records per chunk for streaming
            filters: Query filters to apply
            columns: Specific columns to export
        """
        self.format = format
        self.compression = compression
        self.include_metadata = include_metadata
        self.chunk_size = chunk_size
        self.filters = filters or {}
        self.columns = columns


class DataExporter:
    """Advanced data export engine with streaming and scheduling support."""
    
    def __init__(self, analytics_warehouse: AnalyticsDataWarehouse,
                 db_adapter: DatabaseAdapter = None):
        """Initialize the data exporter.
        
        Args:
            analytics_warehouse: Analytics data warehouse
            db_adapter: Main database adapter for article exports
        """
        self.analytics_warehouse = analytics_warehouse
        self.db_adapter = db_adapter
        
        # Setup directories
        self.exports_dir = Path("analytics/exports")
        self.temp_dir = Path("analytics/exports/temp")
        self.scheduled_exports_dir = Path("analytics/scheduled_exports")
        
        for directory in [self.exports_dir, self.temp_dir, self.scheduled_exports_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def export_data(self, data_type: str, config: ExportConfig,
                   output_file: str = None, stream: bool = False) -> Dict[str, Any]:
        """Export data with specified configuration.
        
        Args:
            data_type: Type of data to export (articles, metrics, etc.)
            config: Export configuration
            output_file: Optional output file path
            stream: Whether to use streaming for large datasets
            
        Returns:
            Export results with file information
        """
        try:
            # Generate output filename if not provided
            if not output_file:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                extension = self._get_file_extension(config.format, config.compression)
                output_file = self.exports_dir / f"{data_type}_{timestamp}.{extension}"
            else:
                output_file = Path(output_file)
            
            self.logger.info(f"Starting {data_type} export to {config.format} format")
            
            # Choose export method based on data type
            if data_type == 'articles':
                result = self._export_articles(config, output_file, stream)
            elif data_type == 'article_metrics':
                result = self._export_article_metrics(config, output_file, stream)
            elif data_type == 'source_metrics':
                result = self._export_source_metrics(config, output_file, stream)
            elif data_type == 'keyword_metrics':
                result = self._export_keyword_metrics(config, output_file, stream)
            elif data_type == 'trend_analysis':
                result = self._export_trend_analysis(config, output_file, stream)
            elif data_type == 'user_activity':
                result = self._export_user_activity(config, output_file, stream)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            # Add common metadata
            result.update({
                'data_type': data_type,
                'format': config.format,
                'compressed': config.compression,
                'exported_at': datetime.utcnow().isoformat(),
                'filters_applied': config.filters
            })
            
            self.logger.info(f"Export completed: {result['file_path']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return {'error': str(e)}
    
    def export_custom_query(self, query: str, params: List[Any],
                          config: ExportConfig, output_file: str = None) -> Dict[str, Any]:
        """Export results of a custom SQL query.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            config: Export configuration
            output_file: Optional output file path
            
        Returns:
            Export results
        """
        try:
            if not output_file:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                extension = self._get_file_extension(config.format, config.compression)
                output_file = self.exports_dir / f"custom_query_{timestamp}.{extension}"
            else:
                output_file = Path(output_file)
            
            # Execute query and get results
            with self.analytics_warehouse.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            # Apply column filter if specified
            if config.columns:
                available_cols = [col for col in config.columns if col in df.columns]
                df = df[available_cols]
            
            # Export data
            record_count = len(df)
            self._export_dataframe(df, config, output_file)
            
            return {
                'success': True,
                'file_path': str(output_file),
                'file_size': output_file.stat().st_size,
                'record_count': record_count,
                'query': query
            }
            
        except Exception as e:
            self.logger.error(f"Custom query export failed: {e}")
            return {'error': str(e)}
    
    def bulk_export(self, export_configs: List[Dict[str, Any]],
                   output_dir: str = None) -> Dict[str, Any]:
        """Perform bulk export of multiple datasets.
        
        Args:
            export_configs: List of export configurations
            output_dir: Output directory for all exports
            
        Returns:
            Bulk export results
        """
        try:
            if not output_dir:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                output_dir = self.exports_dir / f"bulk_export_{timestamp}"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results = []
            successful = 0
            failed = 0
            
            for export_config in export_configs:
                data_type = export_config.get('data_type')
                config = ExportConfig(
                    format=export_config.get('format', ExportFormat.CSV),
                    compression=export_config.get('compression', False),
                    filters=export_config.get('filters', {}),
                    columns=export_config.get('columns')
                )
                
                # Generate output file in bulk directory
                filename = f"{data_type}.{self._get_file_extension(config.format, config.compression)}"
                output_file = output_dir / filename
                
                # Perform export
                result = self.export_data(data_type, config, output_file)
                
                if 'error' in result:
                    failed += 1
                else:
                    successful += 1
                
                results.append({
                    'data_type': data_type,
                    'result': result
                })
            
            # Create manifest file
            manifest = {
                'export_timestamp': datetime.utcnow().isoformat(),
                'total_exports': len(export_configs),
                'successful': successful,
                'failed': failed,
                'exports': results
            }
            
            manifest_file = output_dir / 'manifest.json'
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Create zip archive if requested
            if any(c.get('create_archive', False) for c in export_configs):
                archive_path = self._create_archive(output_dir)
                return {
                    'success': True,
                    'archive_path': str(archive_path),
                    'manifest': manifest
                }
            
            return {
                'success': True,
                'output_dir': str(output_dir),
                'manifest': manifest
            }
            
        except Exception as e:
            self.logger.error(f"Bulk export failed: {e}")
            return {'error': str(e)}
    
    def stream_export(self, data_type: str, config: ExportConfig,
                     callback=None) -> Iterator[bytes]:
        """Stream export data in chunks for large datasets.
        
        Args:
            data_type: Type of data to export
            config: Export configuration
            callback: Optional callback for progress updates
            
        Yields:
            Data chunks as bytes
        """
        try:
            # Create temporary file for streaming
            temp_file = tempfile.NamedTemporaryFile(
                mode='w+b',
                dir=self.temp_dir,
                delete=False,
                suffix=f'.{config.format}'
            )
            temp_path = Path(temp_file.name)
            
            # Get data iterator based on type
            if data_type == 'articles':
                data_iterator = self._stream_articles(config)
            elif data_type == 'article_metrics':
                data_iterator = self._stream_article_metrics(config)
            else:
                raise ValueError(f"Streaming not supported for: {data_type}")
            
            # Format-specific streaming
            if config.format == ExportFormat.CSV:
                yield from self._stream_csv(data_iterator, config, callback)
            elif config.format == ExportFormat.JSONL:
                yield from self._stream_jsonl(data_iterator, config, callback)
            else:
                # For other formats, build complete file first
                self._export_streaming_data(data_iterator, config, temp_path)
                
                # Stream the file contents
                with open(temp_path, 'rb') as f:
                    while chunk := f.read(config.chunk_size):
                        yield chunk
                        if callback:
                            callback({'bytes_sent': len(chunk)})
            
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()
                
        except Exception as e:
            self.logger.error(f"Stream export failed: {e}")
            yield json.dumps({'error': str(e)}).encode()
    
    def _export_articles(self, config: ExportConfig, output_file: Path, 
                        stream: bool = False) -> Dict[str, Any]:
        """Export articles from main database."""
        if not self.db_adapter:
            return {'error': 'Database adapter not available for article exports'}
        
        try:
            # Build query with filters
            query = "SELECT * FROM articles WHERE 1=1"
            params = []
            
            if 'start_date' in config.filters:
                query += " AND published_date >= %s"
                params.append(config.filters['start_date'])
            
            if 'end_date' in config.filters:
                query += " AND published_date <= %s"
                params.append(config.filters['end_date'])
            
            if 'sources' in config.filters:
                query += " AND source = ANY(%s)"
                params.append(config.filters['sources'])
            
            query += " ORDER BY published_date DESC"
            
            # Get data
            with self.db_adapter.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            # Apply column filter
            if config.columns:
                available_cols = [col for col in config.columns if col in df.columns]
                df = df[available_cols]
            
            # Export
            record_count = len(df)
            self._export_dataframe(df, config, output_file)
            
            return {
                'success': True,
                'file_path': str(output_file),
                'file_size': output_file.stat().st_size,
                'record_count': record_count
            }
            
        except Exception as e:
            self.logger.error(f"Failed to export articles: {e}")
            return {'error': str(e)}
    
    def _export_article_metrics(self, config: ExportConfig, output_file: Path,
                               stream: bool = False) -> Dict[str, Any]:
        """Export article metrics from analytics warehouse."""
        try:
            # Build query
            query = """
                SELECT * FROM analytics.article_metrics 
                WHERE time >= %s AND time <= %s
            """
            
            params = [
                config.filters.get('start_time', datetime.utcnow() - timedelta(days=30)),
                config.filters.get('end_time', datetime.utcnow())
            ]
            
            if 'sources' in config.filters:
                query += " AND source = ANY(%s)"
                params.append(config.filters['sources'])
            
            query += " ORDER BY time DESC"
            
            if stream:
                # Use streaming for large datasets
                return self._export_streaming_query(query, params, config, output_file)
            else:
                # Load all data at once
                with self.analytics_warehouse.get_connection() as conn:
                    df = pd.read_sql_query(query, conn, params=params)
                
                # Apply column filter
                if config.columns:
                    available_cols = [col for col in config.columns if col in df.columns]
                    df = df[available_cols]
                
                record_count = len(df)
                self._export_dataframe(df, config, output_file)
                
                return {
                    'success': True,
                    'file_path': str(output_file),
                    'file_size': output_file.stat().st_size,
                    'record_count': record_count
                }
                
        except Exception as e:
            self.logger.error(f"Failed to export article metrics: {e}")
            return {'error': str(e)}
    
    def _export_source_metrics(self, config: ExportConfig, output_file: Path,
                              stream: bool = False) -> Dict[str, Any]:
        """Export source metrics from analytics warehouse."""
        try:
            query = """
                SELECT * FROM analytics.source_metrics
                WHERE time >= %s AND time <= %s
                ORDER BY time DESC, source
            """
            
            params = [
                config.filters.get('start_time', datetime.utcnow() - timedelta(days=30)),
                config.filters.get('end_time', datetime.utcnow())
            ]
            
            with self.analytics_warehouse.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            # Apply filters
            if 'sources' in config.filters:
                df = df[df['source'].isin(config.filters['sources'])]
            
            # Apply column filter
            if config.columns:
                available_cols = [col for col in config.columns if col in df.columns]
                df = df[available_cols]
            
            record_count = len(df)
            self._export_dataframe(df, config, output_file)
            
            return {
                'success': True,
                'file_path': str(output_file),
                'file_size': output_file.stat().st_size,
                'record_count': record_count
            }
            
        except Exception as e:
            self.logger.error(f"Failed to export source metrics: {e}")
            return {'error': str(e)}
    
    def _export_keyword_metrics(self, config: ExportConfig, output_file: Path,
                               stream: bool = False) -> Dict[str, Any]:
        """Export keyword metrics from analytics warehouse."""
        try:
            query = """
                SELECT * FROM analytics.keyword_metrics
                WHERE time >= %s AND time <= %s
            """
            
            params = [
                config.filters.get('start_time', datetime.utcnow() - timedelta(days=30)),
                config.filters.get('end_time', datetime.utcnow())
            ]
            
            if 'keywords' in config.filters:
                query += " AND keyword = ANY(%s)"
                params.append(config.filters['keywords'])
            
            if 'min_mentions' in config.filters:
                query += " AND mention_count >= %s"
                params.append(config.filters['min_mentions'])
            
            query += " ORDER BY time DESC, mention_count DESC"
            
            with self.analytics_warehouse.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            # Apply column filter
            if config.columns:
                available_cols = [col for col in config.columns if col in df.columns]
                df = df[available_cols]
            
            record_count = len(df)
            self._export_dataframe(df, config, output_file)
            
            return {
                'success': True,
                'file_path': str(output_file),
                'file_size': output_file.stat().st_size,
                'record_count': record_count
            }
            
        except Exception as e:
            self.logger.error(f"Failed to export keyword metrics: {e}")
            return {'error': str(e)}
    
    def _export_trend_analysis(self, config: ExportConfig, output_file: Path,
                              stream: bool = False) -> Dict[str, Any]:
        """Export trend analysis results."""
        try:
            query = """
                SELECT * FROM analytics.trend_analysis
                WHERE created_at >= %s AND created_at <= %s
            """
            
            params = [
                config.filters.get('start_time', datetime.utcnow() - timedelta(days=90)),
                config.filters.get('end_time', datetime.utcnow())
            ]
            
            if 'analysis_types' in config.filters:
                query += " AND analysis_type = ANY(%s)"
                params.append(config.filters['analysis_types'])
            
            query += " ORDER BY created_at DESC"
            
            with self.analytics_warehouse.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            # Apply column filter
            if config.columns:
                available_cols = [col for col in config.columns if col in df.columns]
                df = df[available_cols]
            
            record_count = len(df)
            self._export_dataframe(df, config, output_file)
            
            return {
                'success': True,
                'file_path': str(output_file),
                'file_size': output_file.stat().st_size,
                'record_count': record_count
            }
            
        except Exception as e:
            self.logger.error(f"Failed to export trend analysis: {e}")
            return {'error': str(e)}
    
    def _export_user_activity(self, config: ExportConfig, output_file: Path,
                             stream: bool = False) -> Dict[str, Any]:
        """Export user activity data."""
        try:
            query = """
                SELECT * FROM analytics.user_activity
                WHERE time >= %s AND time <= %s
            """
            
            params = [
                config.filters.get('start_time', datetime.utcnow() - timedelta(days=30)),
                config.filters.get('end_time', datetime.utcnow())
            ]
            
            if 'user_ids' in config.filters:
                query += " AND user_id = ANY(%s)"
                params.append(config.filters['user_ids'])
            
            if 'organization_ids' in config.filters:
                query += " AND organization_id = ANY(%s)"
                params.append(config.filters['organization_ids'])
            
            if 'action_types' in config.filters:
                query += " AND action_type = ANY(%s)"
                params.append(config.filters['action_types'])
            
            query += " ORDER BY time DESC"
            
            with self.analytics_warehouse.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            # Apply column filter
            if config.columns:
                available_cols = [col for col in config.columns if col in df.columns]
                df = df[available_cols]
            
            # Anonymize if requested
            if config.filters.get('anonymize', False):
                df = self._anonymize_user_data(df)
            
            record_count = len(df)
            self._export_dataframe(df, config, output_file)
            
            return {
                'success': True,
                'file_path': str(output_file),
                'file_size': output_file.stat().st_size,
                'record_count': record_count,
                'anonymized': config.filters.get('anonymize', False)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to export user activity: {e}")
            return {'error': str(e)}
    
    def _export_dataframe(self, df: pd.DataFrame, config: ExportConfig, output_file: Path):
        """Export DataFrame to specified format.
        
        Args:
            df: DataFrame to export
            config: Export configuration
            output_file: Output file path
        """
        metadata = self._prepare_export_metadata(df, config)
        
        # Use format-specific export methods
        export_methods = {
            ExportFormat.CSV: self._export_to_csv,
            ExportFormat.TSV: self._export_to_tsv,
            ExportFormat.JSON: self._export_to_json,
            ExportFormat.JSONL: self._export_to_jsonl,
            ExportFormat.EXCEL: self._export_to_excel,
            ExportFormat.PARQUET: self._export_to_parquet,
            ExportFormat.AVRO: self._export_to_avro,
            ExportFormat.XML: self._export_to_xml,
            ExportFormat.SQL: self._export_to_sql
        }
        
        if config.format not in export_methods:
            raise ValueError(f"Unsupported export format: {config.format}")
        
        # Execute format-specific export
        export_methods[config.format](df, output_file, metadata, config)
        
        # Apply compression if requested
        if config.compression:
            self._compress_file(output_file)
    
    def _prepare_export_metadata(self, df: pd.DataFrame, config: ExportConfig) -> Optional[Dict[str, Any]]:
        """Prepare export metadata if requested.
        
        Args:
            df: DataFrame being exported
            config: Export configuration
            
        Returns:
            Metadata dictionary or None
        """
        if not config.include_metadata:
            return None
            
        return {
            'exported_at': datetime.utcnow().isoformat(),
            'record_count': len(df),
            'columns': list(df.columns),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
    
    def _export_to_csv(self, df: pd.DataFrame, output_file: Path, 
                      metadata: Optional[Dict[str, Any]], config: ExportConfig):
        """Export DataFrame to CSV format."""
        df.to_csv(output_file, index=False, encoding='utf-8')
    
    def _export_to_tsv(self, df: pd.DataFrame, output_file: Path,
                      metadata: Optional[Dict[str, Any]], config: ExportConfig):
        """Export DataFrame to TSV format."""
        df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
    
    def _export_to_json(self, df: pd.DataFrame, output_file: Path,
                       metadata: Optional[Dict[str, Any]], config: ExportConfig):
        """Export DataFrame to JSON format."""
        if metadata:
            output_data = {
                'metadata': metadata,
                'data': df.to_dict('records')
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, default=str)
        else:
            df.to_json(output_file, orient='records', indent=2, default_handler=str)
    
    def _export_to_jsonl(self, df: pd.DataFrame, output_file: Path,
                        metadata: Optional[Dict[str, Any]], config: ExportConfig):
        """Export DataFrame to JSONL format."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                json.dump(row.to_dict(), f, default=str)
                f.write('\n')
    
    def _export_to_excel(self, df: pd.DataFrame, output_file: Path,
                        metadata: Optional[Dict[str, Any]], config: ExportConfig):
        """Export DataFrame to Excel format."""
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            
            # Add metadata sheet if requested
            if metadata:
                metadata_df = pd.DataFrame([metadata])
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            # Auto-adjust column widths
            self._adjust_excel_column_widths(writer, df)
    
    def _export_to_parquet(self, df: pd.DataFrame, output_file: Path,
                          metadata: Optional[Dict[str, Any]], config: ExportConfig):
        """Export DataFrame to Parquet format."""
        if metadata:
            table = pa.Table.from_pandas(df)
            table = table.replace_schema_metadata({'export_metadata': json.dumps(metadata)})
            pq.write_table(table, output_file)
        else:
            df.to_parquet(output_file, index=False)
    
    def _export_to_avro(self, df: pd.DataFrame, output_file: Path,
                       metadata: Optional[Dict[str, Any]], config: ExportConfig):
        """Export DataFrame to Avro format."""
        if not AVRO_AVAILABLE:
            raise ImportError("Avro support not available. Install 'avro-python3'")
        
        schema = self._create_avro_schema(df)
        
        with open(output_file, 'wb') as f:
            writer = avro.datafile.DataFileWriter(f, avro.io.DatumWriter(), schema)
            
            for _, row in df.iterrows():
                writer.append(row.to_dict())
            
            writer.close()
    
    def _export_to_xml(self, df: pd.DataFrame, output_file: Path,
                      metadata: Optional[Dict[str, Any]], config: ExportConfig):
        """Export DataFrame to XML format."""
        df.to_xml(output_file, index=False, root_name='data', row_name='record')
    
    def _export_to_sql(self, df: pd.DataFrame, output_file: Path,
                      metadata: Optional[Dict[str, Any]], config: ExportConfig):
        """Export DataFrame to SQL format."""
        table_name = output_file.stem
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"-- Generated SQL export\n")
            f.write(f"-- Table: {table_name}\n")
            f.write(f"-- Records: {len(df)}\n\n")
            
            # Write INSERT statements
            columns = ', '.join(df.columns)
            for _, row in df.iterrows():
                values = ', '.join([
                    f"'{str(v).replace('\'', '\'\'')}'" if pd.notna(v) else 'NULL' 
                    for v in row.values
                ])
                f.write(f"INSERT INTO {table_name} ({columns}) VALUES ({values});\n")
    
    def _adjust_excel_column_widths(self, writer: pd.ExcelWriter, df: pd.DataFrame):
        """Adjust Excel column widths for better readability."""
        workbook = writer.book
        worksheet = writer.sheets['Data']
        
        for i, col in enumerate(df.columns):
            column_width = max(df[col].astype(str).map(len).max(), len(str(col))) + 2
            worksheet.set_column(i, i, min(column_width, 50))
    
    def _export_streaming_query(self, query: str, params: List[Any],
                               config: ExportConfig, output_file: Path) -> Dict[str, Any]:
        """Export large query results using streaming."""
        try:
            record_count = 0
            
            with self.analytics_warehouse.get_connection() as conn:
                cursor = conn.cursor('export_cursor')
                cursor.itersize = config.chunk_size
                cursor.execute(query, params)
                
                # Initialize output file based on format
                if config.format == ExportFormat.CSV:
                    with open(output_file, 'w', newline='', encoding='utf-8') as f:
                        writer = None
                        
                        for chunk in self._fetch_chunks(cursor, config.chunk_size):
                            df_chunk = pd.DataFrame(chunk, columns=[desc[0] for desc in cursor.description])
                            
                            if writer is None:
                                writer = csv.DictWriter(f, fieldnames=df_chunk.columns)
                                writer.writeheader()
                            
                            for _, row in df_chunk.iterrows():
                                writer.writerow(row.to_dict())
                            
                            record_count += len(df_chunk)
                
                elif config.format == ExportFormat.JSONL:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        for chunk in self._fetch_chunks(cursor, config.chunk_size):
                            df_chunk = pd.DataFrame(chunk, columns=[desc[0] for desc in cursor.description])
                            
                            for _, row in df_chunk.iterrows():
                                json.dump(row.to_dict(), f, default=str)
                                f.write('\n')
                            
                            record_count += len(df_chunk)
                
                else:
                    # For other formats, collect all data first
                    all_data = []
                    for chunk in self._fetch_chunks(cursor, config.chunk_size):
                        all_data.extend(chunk)
                        record_count += len(chunk)
                    
                    df = pd.DataFrame(all_data, columns=[desc[0] for desc in cursor.description])
                    self._export_dataframe(df, config, output_file)
            
            return {
                'success': True,
                'file_path': str(output_file),
                'file_size': output_file.stat().st_size,
                'record_count': record_count,
                'streaming': True
            }
            
        except Exception as e:
            self.logger.error(f"Streaming export failed: {e}")
            return {'error': str(e)}
    
    def _fetch_chunks(self, cursor, chunk_size: int) -> Iterator[List[tuple]]:
        """Fetch data in chunks from database cursor."""
        while True:
            chunk = cursor.fetchmany(chunk_size)
            if not chunk:
                break
            yield chunk
    
    def _stream_articles(self, config: ExportConfig) -> Iterator[Dict[str, Any]]:
        """Stream articles from database."""
        query = "SELECT * FROM articles WHERE 1=1"
        params = []
        
        if 'start_date' in config.filters:
            query += " AND published_date >= %s"
            params.append(config.filters['start_date'])
        
        if 'end_date' in config.filters:
            query += " AND published_date <= %s"
            params.append(config.filters['end_date'])
        
        query += " ORDER BY published_date DESC"
        
        with self.db_adapter.get_connection() as conn:
            cursor = conn.cursor('stream_cursor')
            cursor.itersize = config.chunk_size
            cursor.execute(query, params)
            
            columns = [desc[0] for desc in cursor.description]
            
            for row in cursor:
                yield dict(zip(columns, row))
    
    def _stream_article_metrics(self, config: ExportConfig) -> Iterator[Dict[str, Any]]:
        """Stream article metrics from analytics warehouse."""
        query = """
            SELECT * FROM analytics.article_metrics 
            WHERE time >= %s AND time <= %s
            ORDER BY time DESC
        """
        
        params = [
            config.filters.get('start_time', datetime.utcnow() - timedelta(days=30)),
            config.filters.get('end_time', datetime.utcnow())
        ]
        
        with self.analytics_warehouse.get_connection() as conn:
            cursor = conn.cursor('stream_cursor')
            cursor.itersize = config.chunk_size
            cursor.execute(query, params)
            
            columns = [desc[0] for desc in cursor.description]
            
            for row in cursor:
                yield dict(zip(columns, row))
    
    def _stream_csv(self, data_iterator: Iterator[Dict], config: ExportConfig,
                   callback=None) -> Iterator[bytes]:
        """Stream data as CSV format."""
        buffer = io.StringIO()
        writer = None
        records_processed = 0
        
        for record in data_iterator:
            if writer is None:
                # Initialize CSV writer with headers
                writer = csv.DictWriter(buffer, fieldnames=record.keys())
                writer.writeheader()
                yield buffer.getvalue().encode('utf-8')
                buffer.seek(0)
                buffer.truncate(0)
            
            # Write record
            writer.writerow(record)
            records_processed += 1
            
            # Yield buffer content when it reaches chunk size
            if buffer.tell() >= config.chunk_size:
                yield buffer.getvalue().encode('utf-8')
                buffer.seek(0)
                buffer.truncate(0)
                
                if callback:
                    callback({'records_processed': records_processed})
        
        # Yield remaining content
        if buffer.tell() > 0:
            yield buffer.getvalue().encode('utf-8')
    
    def _stream_jsonl(self, data_iterator: Iterator[Dict], config: ExportConfig,
                     callback=None) -> Iterator[bytes]:
        """Stream data as JSON Lines format."""
        records_processed = 0
        buffer = []
        buffer_size = 0
        
        for record in data_iterator:
            # Convert record to JSON line
            line = json.dumps(record, default=str) + '\n'
            line_bytes = line.encode('utf-8')
            
            buffer.append(line_bytes)
            buffer_size += len(line_bytes)
            records_processed += 1
            
            # Yield buffer when it reaches chunk size
            if buffer_size >= config.chunk_size:
                yield b''.join(buffer)
                buffer = []
                buffer_size = 0
                
                if callback:
                    callback({'records_processed': records_processed})
        
        # Yield remaining content
        if buffer:
            yield b''.join(buffer)
    
    def _compress_file(self, file_path: Path):
        """Compress file using gzip."""
        compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
        
        with open(file_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                f_out.writelines(f_in)
        
        # Remove original file and rename compressed
        file_path.unlink()
        compressed_path.rename(file_path.with_suffix(file_path.suffix + '.gz'))
    
    def _create_archive(self, directory: Path) -> Path:
        """Create zip archive of directory."""
        archive_path = directory.with_suffix('.zip')
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(directory)
                    zipf.write(file_path, arcname)
        
        return archive_path
    
    def _anonymize_user_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Anonymize sensitive user data."""
        import hashlib
        
        # Hash user IDs
        if 'user_id' in df.columns:
            df['user_id'] = df['user_id'].apply(
                lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
            )
        
        # Remove or mask sensitive columns
        sensitive_columns = ['ip_address', 'user_agent', 'email']
        for col in sensitive_columns:
            if col in df.columns:
                if col == 'ip_address':
                    # Keep only first two octets
                    df[col] = df[col].apply(
                        lambda x: '.'.join(str(x).split('.')[:2]) + '.x.x' if pd.notna(x) else None
                    )
                else:
                    df[col] = 'REDACTED'
        
        return df
    
    def _create_avro_schema(self, df: pd.DataFrame) -> dict:
        """Create Avro schema from DataFrame."""
        fields = []
        
        for col, dtype in df.dtypes.items():
            if dtype == 'int64':
                avro_type = 'long'
            elif dtype == 'float64':
                avro_type = 'double'
            elif dtype == 'bool':
                avro_type = 'boolean'
            else:
                avro_type = 'string'
            
            fields.append({
                'name': col,
                'type': ['null', avro_type],
                'default': None
            })
        
        return {
            'namespace': 'mimir.analytics',
            'type': 'record',
            'name': 'ExportRecord',
            'fields': fields
        }
    
    def _get_file_extension(self, format: str, compression: bool) -> str:
        """Get file extension based on format and compression."""
        extensions = {
            ExportFormat.CSV: 'csv',
            ExportFormat.TSV: 'tsv',
            ExportFormat.JSON: 'json',
            ExportFormat.JSONL: 'jsonl',
            ExportFormat.EXCEL: 'xlsx',
            ExportFormat.PARQUET: 'parquet',
            ExportFormat.AVRO: 'avro',
            ExportFormat.XML: 'xml',
            ExportFormat.SQL: 'sql'
        }
        
        ext = extensions.get(format, 'dat')
        
        if compression:
            ext += '.gz'
        
        return ext