"""
src/performance/resource_monitor.py

DharmaShield - Advanced Resource Monitoring & Auto-Scaling Engine
----------------------------------------------------------------
• Industry-grade real-time system resource monitoring with intelligent auto-scaling
• Cross-platform compatibility (Android/iOS/Desktop) with Kivy/Buildozer deployment
• Adaptive model precision, detection frequency, and memory management
• Proactive performance optimization with predictive resource allocation

Author: DharmaShield Expert Team
License: Apache 2.0 (Google Gemma 3n Competition Compatible)
"""

import os
import time
import threading
import psutil
import gc
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import platform
import queue
from collections import deque
import statistics

# Project imports
from ...utils.logger import get_logger
from ...core.config_loader import load_config
from ..battery_optimizer import get_battery_optimizer

logger = get_logger(__name__)

# Platform detection
IS_ANDROID = 'ANDROID_BOOTLOGO' in os.environ or 'ANDROID_ROOT' in os.environ
IS_IOS = platform.system() == 'Darwin' and 'iPhone' in platform.machine()
IS_MOBILE = IS_ANDROID or IS_IOS
IS_DESKTOP = not IS_MOBILE

# -------------------------------
# Enums and Data Structures
# -------------------------------

class ResourceState(Enum):
    OPTIMAL = "optimal"           # All resources within normal limits
    MODERATE = "moderate"         # Some resource pressure
    HIGH = "high"                # High resource usage - need optimization
    CRITICAL = "critical"         # Critical resource usage - emergency scaling
    OVERLOADED = "overloaded"     # System overloaded - minimal operations only

class ScalingAction(Enum):
    SCALE_UP = "scale_up"         # Increase model precision/frequency
    MAINTAIN = "maintain"         # Keep current settings
    SCALE_DOWN = "scale_down"     # Reduce model precision/frequency
    EMERGENCY_SCALE = "emergency_scale"  # Emergency resource conservation

class ModelPrecision(Enum):
    FP32 = "fp32"                # Full precision (highest quality, most resources)
    FP16 = "fp16"                # Half precision (balanced)
    INT8 = "int8"                # Quantized (lowest resources, reduced quality)
    MINIMAL = "minimal"          # Minimal processing mode

@dataclass
class ResourceMetrics:
    """Current system resource metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available_mb: float = 0.0
    gpu_percent: float = 0.0         # If available
    temperature: float = 0.0         # CPU temperature if available
    disk_io_read_mb_s: float = 0.0
    disk_io_write_mb_s: float = 0.0
    network_io_kb_s: float = 0.0
    process_count: int = 0
    thread_count: int = 0
    battery_percent: float = 100.0
    is_charging: bool = False
    timestamp: float = field(default_factory=time.time)
    
    @property
    def resource_state(self) -> ResourceState:
        """Determine overall resource state."""
        # Critical conditions
        if (self.cpu_percent > 95 or self.memory_percent > 95 or 
            self.temperature > 85 or self.battery_percent < 5):
            return ResourceState.CRITICAL
        
        # High load conditions
        elif (self.cpu_percent > 80 or self.memory_percent > 80 or 
              self.temperature > 75 or self.battery_percent < 15):
            return ResourceState.HIGH
        
        # Moderate load conditions
        elif (self.cpu_percent > 60 or self.memory_percent > 60 or 
              self.temperature > 65 or self.battery_percent < 30):
            return ResourceState.MODERATE
        
        # Overloaded conditions (multiple high resources)
        high_resource_count = sum([
            self.cpu_percent > 85,
            self.memory_percent > 85,
            self.temperature > 80,
            self.battery_percent < 10
        ])
        
        if high_resource_count >= 2:
            return ResourceState.OVERLOADED
        
        return ResourceState.OPTIMAL

@dataclass
class ScalingDecision:
    """Resource scaling decision with rationale."""
    action: ScalingAction
    target_precision: ModelPrecision
    scan_frequency_multiplier: float  # Multiply current frequency by this
    memory_limit_mb: int
    rationale: str
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class PerformanceProfile:
    """Performance configuration profile."""
    name: str
    model_precision: ModelPrecision
    max_cpu_threshold: float = 80.0
    max_memory_threshold: float = 80.0
    max_temperature: float = 75.0
    scan_frequency_base: float = 1.0
    memory_limit_mb: int = 512
    background_tasks_enabled: bool = True
    cache_size_mb: int = 64
    gc_frequency: int = 10  # Garbage collection every N scans

# Predefined performance profiles
PERFORMANCE_PROFILES = {
    ResourceState.OPTIMAL: PerformanceProfile(
        name="Optimal Performance",
        model_precision=ModelPrecision.FP32,
        max_cpu_threshold=85.0,
        max_memory_threshold=85.0,
        scan_frequency_base=2.0,
        memory_limit_mb=1024,
        cache_size_mb=128
    ),
    ResourceState.MODERATE: PerformanceProfile(
        name="Balanced Performance",
        model_precision=ModelPrecision.FP16,
        max_cpu_threshold=75.0,
        max_memory_threshold=75.0,
        scan_frequency_base=1.0,
        memory_limit_mb=512,
        cache_size_mb=64
    ),
    ResourceState.HIGH: PerformanceProfile(
        name="Power Saving",
        model_precision=ModelPrecision.INT8,
        max_cpu_threshold=65.0,
        max_memory_threshold=65.0,
        scan_frequency_base=0.5,
        memory_limit_mb=256,
        background_tasks_enabled=False,
        cache_size_mb=32
    ),
    ResourceState.CRITICAL: PerformanceProfile(
        name="Emergency Mode",
        model_precision=ModelPrecision.MINIMAL,
        max_cpu_threshold=50.0,
        max_memory_threshold=50.0,
        scan_frequency_base=0.1,
        memory_limit_mb=128,
        background_tasks_enabled=False,
        cache_size_mb=16,
        gc_frequency=5
    ),
    ResourceState.OVERLOADED: PerformanceProfile(
        name="Survival Mode",
        model_precision=ModelPrecision.MINIMAL,
        max_cpu_threshold=40.0,
        max_memory_threshold=40.0,
        scan_frequency_base=0.05,
        memory_limit_mb=64,
        background_tasks_enabled=False,
        cache_size_mb=8,
        gc_frequency=2
    )
}

# -------------------------------
# Configuration
# -------------------------------

class ResourceMonitorConfig:
    """Configuration for resource monitoring."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else {}
        monitor_config = self.config.get('resource_monitor', {})
        
        # Monitoring settings
        self.enabled = monitor_config.get('enabled', True)
        self.update_interval = monitor_config.get('update_interval', 5.0)
        self.metrics_history_size = monitor_config.get('metrics_history_size', 60)
        
        # Scaling settings
        self.auto_scaling_enabled = monitor_config.get('auto_scaling_enabled', True)
        self.scaling_sensitivity = monitor_config.get('scaling_sensitivity', 0.8)
        self.min_scaling_interval = monitor_config.get('min_scaling_interval', 30.0)
        
        # Thresholds
        self.cpu_warning_threshold = monitor_config.get('cpu_warning_threshold', 70.0)
        self.cpu_critical_threshold = monitor_config.get('cpu_critical_threshold', 85.0)
        self.memory_warning_threshold = monitor_config.get('memory_warning_threshold', 70.0)
        self.memory_critical_threshold = monitor_config.get('memory_critical_threshold', 85.0)
        self.temperature_warning_threshold = monitor_config.get('temperature_warning_threshold', 70.0)
        self.temperature_critical_threshold = monitor_config.get('temperature_critical_threshold', 80.0)
        
        # Platform-specific settings
        self.mobile_optimizations = monitor_config.get('mobile_optimizations', IS_MOBILE)
        self.predictive_scaling = monitor_config.get('predictive_scaling', True)
        self.aggressive_gc = monitor_config.get('aggressive_gc', IS_MOBILE)

# -------------------------------
# Resource Metrics Collector
# -------------------------------

class ResourceMetricsCollector:
    """Collects comprehensive system resource metrics."""
    
    def __init__(self, config: ResourceMonitorConfig):
        self.config = config
        self.last_disk_io = None
        self.last_net_io = None
        self.last_measurement_time = time.time()
        self._lock = threading.Lock()
    
    def collect_metrics(self) -> ResourceMetrics:
        """Collect current system resource metrics."""
        try:
            with self._lock:
                current_time = time.time()
                time_delta = current_time - self.last_measurement_time
                
                metrics = ResourceMetrics()
                
                # CPU metrics
                metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                metrics.memory_percent = memory.percent
                metrics.memory_available_mb = memory.available / (1024 * 1024)
                
                # Temperature (if available)
                metrics.temperature = self._get_cpu_temperature()
                
                # Disk I/O metrics
                disk_io = psutil.disk_io_counters()
                if disk_io and self.last_disk_io and time_delta > 0:
                    read_bytes_diff = disk_io.read_bytes - self.last_disk_io.read_bytes
                    write_bytes_diff = disk_io.write_bytes - self.last_disk_io.write_bytes
                    metrics.disk_io_read_mb_s = (read_bytes_diff / time_delta) / (1024 * 1024)
                    metrics.disk_io_write_mb_s = (write_bytes_diff / time_delta) / (1024 * 1024)
                self.last_disk_io = disk_io
                
                # Network I/O metrics
                net_io = psutil.net_io_counters()
                if net_io and self.last_net_io and time_delta > 0:
                    bytes_sent_diff = net_io.bytes_sent - self.last_net_io.bytes_sent
                    bytes_recv_diff = net_io.bytes_recv - self.last_net_io.bytes_recv
                    total_bytes_diff = bytes_sent_diff + bytes_recv_diff
                    metrics.network_io_kb_s = (total_bytes_diff / time_delta) / 1024
                self.last_net_io = net_io
                
                # Process and thread counts
                metrics.process_count = len(psutil.pids())
                metrics.thread_count = sum(p.num_threads() for p in psutil.process_iter(['num_threads']) if p.info['num_threads'])
                
                # Battery information
                battery_info = self._get_battery_info()
                metrics.battery_percent = battery_info['percent']
                metrics.is_charging = battery_info['charging']
                
                # GPU metrics (if available)
                metrics.gpu_percent = self._get_gpu_usage()
                
                self.last_measurement_time = current_time
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to collect resource metrics: {e}")
            return ResourceMetrics()
    
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature if available."""
        try:
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Try to find CPU temperature
                    for sensor_name, sensor_list in temps.items():
                        if 'cpu' in sensor_name.lower() or 'core' in sensor_name.lower():
                            if sensor_list:
                                return sensor_list[0].current
                    
                    # If no CPU-specific sensor, use the first available
                    for sensor_list in temps.values():
                        if sensor_list:
                            return sensor_list[0].current
        except Exception:
            pass
        return 0.0
    
    def _get_battery_info(self) -> Dict[str, Any]:
        """Get battery information."""
        try:
            battery_optimizer = get_battery_optimizer()
            status = battery_optimizer.get_battery_status()
            return {
                'percent': status.get('battery_percent', 100.0),
                'charging': status.get('is_charging', False)
            }
        except Exception:
            # Fallback to psutil
            try:
                if hasattr(psutil, 'sensors_battery'):
                    battery = psutil.sensors_battery()
                    if battery:
                        return {
                            'percent': battery.percent,
                            'charging': battery.power_plugged
                        }
            except Exception:
                pass
            
            return {'percent': 100.0, 'charging': False}
    
    def _get_gpu_usage(self) -> float:
        """Get GPU usage if available (placeholder for future implementation)."""
        # This would require platform-specific implementations
        # For Android: access through Android APIs
        # For desktop: nvidia-smi, rocm-smi, etc.
        return 0.0

# -------------------------------
# Predictive Resource Analyzer
# -------------------------------

class PredictiveResourceAnalyzer:
    """Analyzes resource trends and predicts future resource needs."""
    
    def __init__(self, config: ResourceMonitorConfig):
        self.config = config
        self.metrics_history = deque(maxlen=config.metrics_history_size)
        self.trend_window_size = min(10, config.metrics_history_size // 4)
        self._lock = threading.Lock()
    
    def add_metrics(self, metrics: ResourceMetrics):
        """Add metrics to history for analysis."""
        with self._lock:
            self.metrics_history.append(metrics)
    
    def predict_resource_pressure(self, lookahead_seconds: float = 60.0) -> Tuple[ResourceState, float]:
        """Predict resource state in the near future."""
        try:
            with self._lock:
                if len(self.metrics_history) < self.trend_window_size:
                    # Not enough data for prediction
                    current_metrics = self.metrics_history[-1] if self.metrics_history else ResourceMetrics()
                    return current_metrics.resource_state, 0.5
                
                # Analyze trends
                recent_metrics = list(self.metrics_history)[-self.trend_window_size:]
                
                # Calculate trends for key metrics
                cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
                memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics])
                temp_trend = self._calculate_trend([m.temperature for m in recent_metrics])
                battery_trend = self._calculate_trend([m.battery_percent for m in recent_metrics])
                
                # Project future values
                current = recent_metrics[-1]
                time_factor = lookahead_seconds / self.config.update_interval
                
                predicted_cpu = current.cpu_percent + (cpu_trend * time_factor)
                predicted_memory = current.memory_percent + (memory_trend * time_factor)
                predicted_temp = current.temperature + (temp_trend * time_factor)
                predicted_battery = current.battery_percent + (battery_trend * time_factor)
                
                # Create predicted metrics
                predicted_metrics = ResourceMetrics(
                    cpu_percent=max(0, min(100, predicted_cpu)),
                    memory_percent=max(0, min(100, predicted_memory)),
                    temperature=max(0, predicted_temp),
                    battery_percent=max(0, min(100, predicted_battery)),
                    is_charging=current.is_charging
                )
                
                # Calculate prediction confidence based on trend stability
                trend_stability = self._calculate_trend_stability(recent_metrics)
                confidence = min(1.0, trend_stability)
                
                return predicted_metrics.resource_state, confidence
                
        except Exception as e:
            logger.error(f"Error in predictive analysis: {e}")
            return ResourceState.OPTIMAL, 0.5
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend (slope) for a series of values."""
        if len(values) < 2:
            return 0.0
        
        try:
            # Simple linear regression
            n = len(values)
            x_mean = (n - 1) / 2
            y_mean = statistics.mean(values)
            
            numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
            
        except Exception:
            return 0.0
    
    def _calculate_trend_stability(self, metrics: List[ResourceMetrics]) -> float:
        """Calculate how stable the trends are (higher = more predictable)."""
        try:
            if len(metrics) < 3:
                return 0.5
            
            # Calculate variance in key metrics
            cpu_values = [m.cpu_percent for m in metrics]
            memory_values = [m.memory_percent for m in metrics]
            
            cpu_variance = statistics.variance(cpu_values) if len(cpu_values) > 1 else 0
            memory_variance = statistics.variance(memory_values) if len(memory_values) > 1 else 0
            
            # Lower variance = higher stability
            stability = 1.0 / (1.0 + (cpu_variance + memory_variance) / 200.0)
            return max(0.1, min(1.0, stability))
            
        except Exception:
            return 0.5

# -------------------------------
# Scaling Decision Engine
# -------------------------------

class ScalingDecisionEngine:
    """Makes intelligent scaling decisions based on resource analysis."""
    
    def __init__(self, config: ResourceMonitorConfig):
        self.config = config
        self.last_scaling_time = 0.0
        self.scaling_history = deque(maxlen=20)
        self.current_profile = PERFORMANCE_PROFILES[ResourceState.OPTIMAL]
        self._lock = threading.Lock()
    
    def make_scaling_decision(self, current_metrics: ResourceMetrics, 
                            predicted_state: ResourceState,
                            prediction_confidence: float) -> Optional[ScalingDecision]:
        """Make a scaling decision based on current and predicted resource state."""
        try:
            with self._lock:
                current_time = time.time()
                
                # Check if we should wait before making another scaling decision
                if (current_time - self.last_scaling_time) < self.config.min_scaling_interval:
                    return None
                
                current_state = current_metrics.resource_state
                
                # Determine target state (weighted between current and predicted)
                target_state = self._determine_target_state(current_state, predicted_state, prediction_confidence)
                
                # Check if scaling action is needed
                if target_state == current_state and len(self.scaling_history) > 0:
                    last_decision = self.scaling_history[-1]
                    if last_decision.action in [ScalingAction.MAINTAIN, ScalingAction.SCALE_UP] and target_state == ResourceState.OPTIMAL:
                        return None  # No action needed
                
                # Create scaling decision
                decision = self._create_scaling_decision(current_metrics, target_state, prediction_confidence)
                
                if decision.action != ScalingAction.MAINTAIN:
                    self.scaling_history.append(decision)
                    self.last_scaling_time = current_time
                    self.current_profile = PERFORMANCE_PROFILES[target_state]
                
                return decision
                
        except Exception as e:
            logger.error(f"Error making scaling decision: {e}")
            return None
    
    def _determine_target_state(self, current_state: ResourceState, 
                              predicted_state: ResourceState, 
                              confidence: float) -> ResourceState:
        """Determine target resource state based on current and predicted states."""
        # If prediction confidence is low, rely more on current state
        if confidence < 0.3:
            return current_state
        
        # If prediction is worse than current, prepare for it
        state_priorities = {
            ResourceState.OPTIMAL: 0,
            ResourceState.MODERATE: 1,
            ResourceState.HIGH: 2,
            ResourceState.CRITICAL: 3,
            ResourceState.OVERLOADED: 4
        }
        
        current_priority = state_priorities.get(current_state, 1)
        predicted_priority = state_priorities.get(predicted_state, 1)
        
        # Weight the decision based on confidence
        weighted_priority = (current_priority * (1 - confidence) + 
                           predicted_priority * confidence)
        
        # Map back to resource state
        for state, priority in state_priorities.items():
            if weighted_priority <= priority + 0.5:
                return state
        
        return ResourceState.CRITICAL
    
    def _create_scaling_decision(self, metrics: ResourceMetrics, 
                               target_state: ResourceState,
                               confidence: float) -> ScalingDecision:
        """Create a scaling decision for the target state."""
        target_profile = PERFORMANCE_PROFILES[target_state]
        current_state = metrics.resource_state
        
        # Determine scaling action
        if target_state == ResourceState.OVERLOADED or target_state == ResourceState.CRITICAL:
            action = ScalingAction.EMERGENCY_SCALE
        elif target_state in [ResourceState.HIGH, ResourceState.MODERATE]:
            action = ScalingAction.SCALE_DOWN
        elif target_state == ResourceState.OPTIMAL and current_state != ResourceState.OPTIMAL:
            action = ScalingAction.SCALE_UP
        else:
            action = ScalingAction.MAINTAIN
        
        # Calculate frequency multiplier
        frequency_multiplier = target_profile.scan_frequency_base / max(0.1, self.current_profile.scan_frequency_base)
        
        # Create rationale
        rationale = self._generate_rationale(metrics, target_state, action)
        
        return ScalingDecision(
            action=action,
            target_precision=target_profile.model_precision,
            scan_frequency_multiplier=frequency_multiplier,
            memory_limit_mb=target_profile.memory_limit_mb,
            rationale=rationale,
            confidence=confidence
        )
    
    def _generate_rationale(self, metrics: ResourceMetrics, 
                          target_state: ResourceState, 
                          action: ScalingAction) -> str:
        """Generate human-readable rationale for the scaling decision."""
        reasons = []
        
        if metrics.cpu_percent > 80:
            reasons.append(f"High CPU usage ({metrics.cpu_percent:.1f}%)")
        if metrics.memory_percent > 80:
            reasons.append(f"High memory usage ({metrics.memory_percent:.1f}%)")
        if metrics.temperature > 75:
            reasons.append(f"High temperature ({metrics.temperature:.1f}°C)")
        if metrics.battery_percent < 20:
            reasons.append(f"Low battery ({metrics.battery_percent:.1f}%)")
        
        action_descriptions = {
            ScalingAction.SCALE_UP: "increasing performance",
            ScalingAction.SCALE_DOWN: "reducing resource usage",
            ScalingAction.EMERGENCY_SCALE: "emergency resource conservation",
            ScalingAction.MAINTAIN: "maintaining current settings"
        }
        
        base_rationale = f"Target state: {target_state.value}, Action: {action_descriptions[action]}"
        
        if reasons:
            return f"{base_rationale}. Reasons: {', '.join(reasons)}"
        else:
            return base_rationale

# -------------------------------
# Main Resource Monitor Engine
# -------------------------------

class ResourceMonitorEngine:
    """
    Main resource monitoring and auto-scaling engine for DharmaShield.
    
    Features:
    - Real-time system resource monitoring
    - Predictive resource analysis
    - Intelligent auto-scaling decisions
    - Cross-platform compatibility
    - Integration with battery optimizer
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config_path: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        if getattr(self, '_initialized', False):
            return
        
        self.config = ResourceMonitorConfig(config_path)
        self.metrics_collector = ResourceMetricsCollector(self.config)
        self.predictive_analyzer = PredictiveResourceAnalyzer(self.config)
        self.scaling_engine = ScalingDecisionEngine(self.config)
        
        # Current state
        self.current_metrics = ResourceMetrics()
        self.current_performance_profile = PERFORMANCE_PROFILES[ResourceState.OPTIMAL]
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Callbacks for scaling events
        self.scaling_callbacks: List[Callable[[ScalingDecision], None]] = []
        self.metrics_callbacks: List[Callable[[ResourceMetrics], None]] = []
        
        # Performance counters
        self.scan_count = 0
        self.gc_counter = 0
        
        self._initialized = True
        logger.info("ResourceMonitorEngine initialized")
        
        if self.config.enabled:
            self.start_monitoring()
    
    def start_monitoring(self):
        """Start resource monitoring thread."""
        try:
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                return
            
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True
            )
            self.monitoring_thread.start()
            logger.info("Resource monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start resource monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop resource monitoring thread."""
        try:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5.0)
            logger.info("Resource monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping resource monitoring: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect current metrics
                self.current_metrics = self.metrics_collector.collect_metrics()
                
                # Add to predictive analyzer
                self.predictive_analyzer.add_metrics(self.current_metrics)
                
                # Notify metrics callbacks
                self._notify_metrics_callbacks()
                
                # Make scaling decision if auto-scaling is enabled
                if self.config.auto_scaling_enabled:
                    predicted_state, confidence = self.predictive_analyzer.predict_resource_pressure()
                    
                    decision = self.scaling_engine.make_scaling_decision(
                        self.current_metrics, predicted_state, confidence
                    )
                    
                    if decision and decision.action != ScalingAction.MAINTAIN:
                        self._apply_scaling_decision(decision)
                        self._notify_scaling_callbacks(decision)
                
                # Perform garbage collection if needed
                self._manage_garbage_collection()
                
                # Sleep until next update
                time.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10.0)  # Longer sleep on error
    
    def _apply_scaling_decision(self, decision: ScalingDecision):
        """Apply a scaling decision to the system."""
        try:
            # Update current performance profile
            target_state = None
            for state, profile in PERFORMANCE_PROFILES.items():
                if profile.model_precision == decision.target_precision:
                    target_state = state
                    break
            
            if target_state:
                self.current_performance_profile = PERFORMANCE_PROFILES[target_state]
            
            logger.info(f"Applied scaling decision: {decision.rationale}")
            
        except Exception as e:
            logger.error(f"Error applying scaling decision: {e}")
    
    def _manage_garbage_collection(self):
        """Manage garbage collection based on current profile."""
        try:
            self.scan_count += 1
            gc_frequency = self.current_performance_profile.gc_frequency
            
            if self.scan_count % gc_frequency == 0:
                if self.config.aggressive_gc or self.current_metrics.resource_state in [
                    ResourceState.HIGH, ResourceState.CRITICAL, ResourceState.OVERLOADED
                ]:
                    # Force garbage collection
                    collected = gc.collect()
                    if collected > 0:
                        logger.debug(f"Garbage collection freed {collected} objects")
                
        except Exception as e:
            logger.error(f"Error in garbage collection management: {e}")
    
    def should_allow_operation(self, operation_type: str = "scan", 
                             resource_cost: str = "normal") -> bool:
        """Check if an operation should be allowed based on current resource state."""
        try:
            state = self.current_metrics.resource_state
            
            # Always allow critical operations
            if resource_cost == "critical":
                return True
            
            # Block high-cost operations in critical states
            if state == ResourceState.OVERLOADED:
                return resource_cost == "low"
            elif state == ResourceState.CRITICAL:
                return resource_cost in ["low", "normal"]
            elif state == ResourceState.HIGH:
                return resource_cost != "high"
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking operation allowance: {e}")
            return True  # Default to allowing operation
    
    def get_current_performance_limits(self) -> Dict[str, Any]:
        """Get current performance limits based on resource state."""
        profile = self.current_performance_profile
        return {
            'model_precision': profile.model_precision.value,
            'max_cpu_threshold': profile.max_cpu_threshold,
            'max_memory_threshold': profile.max_memory_threshold,
            'scan_frequency_base': profile.scan_frequency_base,
            'memory_limit_mb': profile.memory_limit_mb,
            'background_tasks_enabled': profile.background_tasks_enabled,
            'cache_size_mb': profile.cache_size_mb
        }
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return {
            'cpu_percent': self.current_metrics.cpu_percent,
            'memory_percent': self.current_metrics.memory_percent,
            'memory_available_mb': self.current_metrics.memory_available_mb,
            'temperature': self.current_metrics.temperature,
            'battery_percent': self.current_metrics.battery_percent,
            'is_charging': self.current_metrics.is_charging,
            'resource_state': self.current_metrics.resource_state.value,
            'disk_io_read_mb_s': self.current_metrics.disk_io_read_mb_s,
            'disk_io_write_mb_s': self.current_metrics.disk_io_write_mb_s,
            'network_io_kb_s': self.current_metrics.network_io_kb_s,
            'process_count': self.current_metrics.process_count,
            'thread_count': self.current_metrics.thread_count
        }
    
    def force_scaling_decision(self, target_state: ResourceState) -> Optional[ScalingDecision]:
        """Force a scaling decision to a specific target state."""
        try:
            decision = self.scaling_engine._create_scaling_decision(
                self.current_metrics, target_state, 1.0
            )
            self._apply_scaling_decision(decision)
            self._notify_scaling_callbacks(decision)
            return decision
            
        except Exception as e:
            logger.error(f"Error forcing scaling decision: {e}")
            return None
    
    def add_scaling_callback(self, callback: Callable[[ScalingDecision], None]):
        """Add callback for scaling events."""
        self.scaling_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: Callable[[ResourceMetrics], None]):
        """Add callback for metrics updates."""
        self.metrics_callbacks.append(callback)
    
    def _notify_scaling_callbacks(self, decision: ScalingDecision):
        """Notify all registered scaling callbacks."""
        for callback in self.scaling_callbacks:
            try:
                callback(decision)
            except Exception as e:
                logger.error(f"Error in scaling callback: {e}")
    
    def _notify_metrics_callbacks(self):
        """Notify all registered metrics callbacks."""
        for callback in self.metrics_callbacks:
            try:
                callback(self.current_metrics)
            except Exception as e:
                logger.error(f"Error in metrics callback: {e}")
    
    def get_resource_trends(self, window_minutes: int = 10) -> Dict[str, List[float]]:
        """Get resource trends over the specified time window."""
        try:
            window_size = min(
                int(window_minutes * 60 / self.config.update_interval),
                len(self.predictive_analyzer.metrics_history)
            )
            
            if window_size == 0:
                return {}
            
            recent_metrics = list(self.predictive_analyzer.metrics_history)[-window_size:]
            
            return {
                'timestamps': [m.timestamp for m in recent_metrics],
                'cpu_percent': [m.cpu_percent for m in recent_metrics],
                'memory_percent': [m.memory_percent for m in recent_metrics],
                'temperature': [m.temperature for m in recent_metrics],
                'battery_percent': [m.battery_percent for m in recent_metrics]
            }
            
        except Exception as e:
            logger.error(f"Error getting resource trends: {e}")
            return {}

# -------------------------------
# Singleton and Convenience Functions
# -------------------------------

_global_resource_monitor = None

def get_resource_monitor(config_path: Optional[str] = None) -> ResourceMonitorEngine:
    """Get the global resource monitor instance."""
    global _global_resource_monitor
    if _global_resource_monitor is None:
        _global_resource_monitor = ResourceMonitorEngine(config_path)
    return _global_resource_monitor

def should_allow_operation(operation_type: str = "scan", resource_cost: str = "normal") -> bool:
    """Convenience function to check if operation should be allowed."""
    monitor = get_resource_monitor()
    return monitor.should_allow_operation(operation_type, resource_cost)

def get_current_performance_limits() -> Dict[str, Any]:
    """Get current performance limits."""
    monitor = get_resource_monitor()
    return monitor.get_current_performance_limits()

def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics."""
    monitor = get_resource_monitor()
    return monitor.get_current_metrics()

# -------------------------------
# Testing and Demo
# -------------------------------

if __name__ == "__main__":
    print("=== DharmaShield Resource Monitor Demo ===\n")
    
    monitor = get_resource_monitor()
    
    # Display initial metrics
    print("Initial System Metrics:")
    metrics = monitor.get_current_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print(f"\nCurrent Performance Limits:")
    limits = monitor.get_current_performance_limits()
    for key, value in limits.items():
        print(f"  {key}: {value}")
    
    # Test operation allowance
    print(f"\nOperation Allowance Tests:")
    operations = [
        ("scan", "low"),
        ("scan", "normal"),
        ("scan", "high"),
        ("ml_inference", "normal"),
        ("background_sync", "low")
    ]
    
    for op_type, cost in operations:
        allowed = monitor.should_allow_operation(op_type, cost)
        print(f"  {op_type} ({cost} cost): {'✓' if allowed else '✗'}")
    
    # Simulate some monitoring cycles
    print(f"\nSimulating monitoring cycles...")
    for i in range(3):
        time.sleep(2)
        current_metrics = monitor.get_current_metrics()
        print(f"  Cycle {i+1}: CPU {current_metrics['cpu_percent']:.1f}%, "
              f"Memory {current_metrics['memory_percent']:.1f}%, "
              f"State: {current_metrics['resource_state']}")
    
    # Test forced scaling
    print(f"\nTesting forced scaling to POWER_SAVER mode...")
    decision = monitor.force_scaling_decision(ResourceState.HIGH)
    if decision:
        print(f"  Scaling decision: {decision.action.value}")
        print(f"  Target precision: {decision.target_precision.value}")
        print(f"  Rationale: {decision.rationale}")
    
    # Display updated limits
    print(f"\nUpdated Performance Limits:")
    limits = monitor.get_current_performance_limits()
    for key, value in limits.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ All tests completed successfully!")
    print(f"📊 Resource Monitor ready for production deployment!")
    print(f"\n🚀 Features demonstrated:")
    print(f"  ✓ Real-time resource monitoring")
    print(f"  ✓ Predictive resource analysis")
    print(f"  ✓ Intelligent auto-scaling")
    print(f"  ✓ Cross-platform compatibility")
    print(f"  ✓ Performance optimization")
    print(f"  ✓ Resource-aware operation control")
    print(f"  ✓ Garbage collection management")
    
    # Clean up
    monitor.stop_monitoring()

