import logging
import asyncio
import psutil
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from config import ConfigManager
from database import DatabaseManager
from exchange import BinanceExchange
from analysis import TechnicalAnalysis
from ml_model import MLModel
from strategy import Strategy

logger = logging.getLogger(__name__)

class MonitoringTasks:
    def __init__(self, config: ConfigManager, db: DatabaseManager,
                 exchange: BinanceExchange, analysis: TechnicalAnalysis,
                 ml_model: MLModel, strategy: Strategy):
        """Initialize monitoring tasks"""
        self.config = config
        self.db = db
        self.exchange = exchange
        self.analysis = analysis
        self.ml_model = ml_model
        self.strategy = strategy
        self.stop_event = asyncio.Event()
        self.logger = logging.getLogger(__name__)
        
        # Monitoring thresholds
        self.cpu_threshold = self.config.monitoring_config.cpu_threshold
        self.memory_threshold = self.config.monitoring_config.memory_threshold
        self.disk_threshold = self.config.monitoring_config.disk_threshold
        self.profit_threshold = self.config.monitoring_config.profit_threshold
        self.drawdown_threshold = self.config.monitoring_config.drawdown_threshold
        
    async def initialize(self):
        """Initialize monitoring tasks"""
        try:
            self.logger.info("Initializing system monitoring...")
            
            # Initialize monitoring
            await self._initialize_monitoring()
            
            # Log configuration
            self.logger.info(f"Monitoring thresholds:")
            self.logger.info(f"CPU usage: {self.cpu_threshold}%")
            self.logger.info(f"Memory usage: {self.memory_threshold}%")
            self.logger.info(f"Disk usage: {self.disk_threshold}%")
            self.logger.info(f"Profit threshold: {self.profit_threshold}%")
            self.logger.info(f"Drawdown threshold: {self.drawdown_threshold}%")
            
            self.logger.info("✅ Monitoring tasks initialized successfully")
            
        except Exception as e:
            error_msg = f"❌ Error initializing monitoring tasks: {str(e)}"
            self.logger.error(error_msg)
            raise
            
    async def start(self):
        """Start monitoring tasks"""
        try:
            self.logger.info("Starting system monitoring...")
            
            # Start monitoring loop
            await self._monitoring_loop()
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring tasks: {e}")
            raise
            
    async def stop(self):
        """Stop monitoring tasks"""
        try:
            self.logger.info("Stopping monitoring tasks...")
            
            # Set stop event
            self.stop_event.set()
            
            self.logger.info("Monitoring tasks stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring tasks: {e}")
            raise
            
    async def update(self):
        """Update monitoring tasks"""
        try:
            # Monitor system resources
            await self._monitor_system()
            
            # Monitor exchange connection
            await self._monitor_exchange()
            
            # Monitor database
            await self._monitor_database()
            
            # Monitor performance
            await self._monitor_performance()
            
        except Exception as e:
            self.logger.error(f"Error updating monitoring tasks: {e}")
            raise
            
    async def _initialize_monitoring(self):
        """Initialize monitoring"""
        try:
            # Initialize system monitoring
            self.system_stats = {
                'cpu_percent': 0,
                'memory_percent': 0,
                'disk_percent': 0,
                'network_io': {'bytes_sent': 0, 'bytes_recv': 0}
            }
            
            # Initialize exchange monitoring
            self.exchange_stats = {
                'connection_status': False,
                'latency': 0,
                'rate_limits': {}
            }
            
            # Initialize database monitoring
            self.database_stats = {
                'connection_status': False,
                'query_time': 0,
                'size': 0
            }
            
            # Initialize performance monitoring
            self.performance_stats = {
                'trades': 0,
                'win_rate': 0,
                'profit_loss': 0,
                'drawdown': 0
            }
            
        except Exception as e:
            self.logger.error(f"Error initializing monitoring: {e}")
            raise
            
    async def _monitoring_loop(self):
        """Monitoring loop"""
        try:
            while not self.stop_event.is_set():
                # Update monitoring
                await self.update()
                
                # Sleep
                await asyncio.sleep(self.config.monitoring_config.update_interval)
                
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
            raise
            
    async def _monitor_system(self):
        """Monitor system resources"""
        try:
            # Get CPU usage
            self.system_stats['cpu_percent'] = psutil.cpu_percent()
            
            # Get memory usage
            memory = psutil.virtual_memory()
            self.system_stats['memory_percent'] = memory.percent
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            self.system_stats['disk_percent'] = disk.percent
            
            # Get network I/O
            network = psutil.net_io_counters()
            self.system_stats['network_io'] = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv
            }
            
            # Log system stats
            self.logger.debug(f"System stats: {self.system_stats}")
            
            # Check thresholds
            if self.system_stats['cpu_percent'] > self.config.monitoring_config.cpu_threshold:
                self.logger.warning(f"High CPU usage: {self.system_stats['cpu_percent']}%")
                
            if self.system_stats['memory_percent'] > self.config.monitoring_config.memory_threshold:
                self.logger.warning(f"High memory usage: {self.system_stats['memory_percent']}%")
                
            if self.system_stats['disk_percent'] > self.config.monitoring_config.disk_threshold:
                self.logger.warning(f"High disk usage: {self.system_stats['disk_percent']}%")
                
        except Exception as e:
            self.logger.error(f"Error monitoring system: {e}")
            raise
            
    async def _monitor_exchange(self):
        """Monitor exchange connection"""
        try:
            # Check connection status
            start_time = time.time()
            connection_status = await self.exchange.check_connection()
            latency = time.time() - start_time
            
            # Get rate limits
            rate_limits = await self.exchange.get_rate_limits()
            
            # Update exchange stats
            self.exchange_stats['connection_status'] = connection_status
            self.exchange_stats['latency'] = latency
            self.exchange_stats['rate_limits'] = rate_limits
            
            # Log exchange stats
            self.logger.debug(f"Exchange stats: {self.exchange_stats}")
            
            # Check thresholds
            if not connection_status:
                self.logger.error("Exchange connection lost")
                
            if latency > self.config.monitoring_config.latency_threshold:
                self.logger.warning(f"High exchange latency: {latency}s")
                
            for limit in rate_limits:
                if limit['used'] > limit['limit'] * 0.9:
                    self.logger.warning(f"Rate limit near threshold: {limit}")
                    
        except Exception as e:
            self.logger.error(f"Error monitoring exchange: {e}")
            raise
            
    async def _monitor_database(self):
        """Monitor database"""
        try:
            # Check connection status
            start_time = time.time()
            connection_status = await self.db.check_connection()
            query_time = time.time() - start_time
            
            # Get database size
            size = await self.db.get_size()
            
            # Update database stats
            self.database_stats['connection_status'] = connection_status
            self.database_stats['query_time'] = query_time
            self.database_stats['size'] = size
            
            # Log database stats
            self.logger.debug(f"Database stats: {self.database_stats}")
            
            # Check thresholds
            if not connection_status:
                self.logger.error("Database connection lost")
                
            if query_time > self.config.monitoring_config.query_threshold:
                self.logger.warning(f"High database query time: {query_time}s")
                
            if size > self.config.monitoring_config.size_threshold:
                self.logger.warning(f"High database size: {size} bytes")
                
        except Exception as e:
            self.logger.error(f"Error monitoring database: {e}")
            raise
            
    async def _monitor_performance(self):
        """Monitor trading performance"""
        try:
            # Get performance metrics
            trades = len(await self.db.get_trades())
            win_rate = await self.db.get_win_rate()
            profit_loss = await self.db.get_profit_loss()
            drawdown = await self.db.get_drawdown()
            
            # Update performance stats
            self.performance_stats['trades'] = trades
            self.performance_stats['win_rate'] = win_rate
            self.performance_stats['profit_loss'] = profit_loss
            self.performance_stats['drawdown'] = drawdown
            
            # Log performance stats
            self.logger.debug(f"Performance stats: {self.performance_stats}")
            
            # Check thresholds
            if win_rate < self.config.monitoring_config.win_rate_threshold:
                self.logger.warning(f"Low win rate: {win_rate}%")
                
            if profit_loss < self.config.monitoring_config.profit_threshold:
                self.logger.warning(f"Low profit: {profit_loss}")
                
            if drawdown > self.config.monitoring_config.drawdown_threshold:
                self.logger.warning(f"High drawdown: {drawdown}%")
                
        except Exception as e:
            self.logger.error(f"Error monitoring performance: {e}")
            raise 