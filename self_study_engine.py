import datetime
import pytz
import json
import os
import random
from typing import List, Dict, Tuple, Optional, Any
import statistics
from collections import defaultdict
import hashlib
import logging
from pathlib import Path

class SelfStudyEngine:
    """Self-study engine for analyzing trading signals and generating insights.
    
    Attributes:
        timezone: The timezone for study window calculations.
        study_window: Tuple of (start_hour, end_hour) for study period.
        log_path: Path to the signal log file.
        memory: List of analyzed signals.
        metrics: Dictionary containing various performance metrics.
    """
    
    def _init_(self, log_path: str = "lae_signal_log.json", study_window: Tuple[int, int] = (2, 7)):
        """Initialize the self-study engine.
        
        Args:
            log_path: Path to the signal log JSON file.
            study_window: Tuple representing (start_hour, end_hour) for study period in IST.
        """
        self.timezone = pytz.timezone("Asia/Kolkata")
        self.study_start, self.study_end = study_window
        self.log_path = log_path
        self.memory: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {
            "total": 0,
            "success": 0,
            "rate": 0.0,
            "conf": [],
            "by_decision": defaultdict(int),
            "by_hour": defaultdict(lambda: defaultdict(int)),
            "by_rr": defaultdict(list)
        }
        self._setup_logging()
        self._load_historical()

    def _setup_logging(self) -> None:
        """Configure basic logging settings."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(_name_)

    def _now_ist(self) -> datetime.datetime:
        """Get current time in IST timezone.
        
        Returns:
            Current datetime in IST timezone.
        """
        return datetime.datetime.now(self.timezone)

    def _in_study_time(self) -> bool:
        """Check if current time is within study window.
        
        Returns:
            True if current hour is within study window, False otherwise.
        """
        hour = self._now_ist().hour
        return self.study_start <= hour < self.study_end

    def _load_signals(self) -> List[Dict[str, Any]]:
        """Load signals from the log file.
        
        Returns:
            List of signal dictionaries or empty list if file not found/invalid.
        """
        if not os.path.exists(self.log_path):
            self.logger.warning(f"Signal log file not found at {self.log_path}")
            return []

        try:
            with open(self.log_path, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Error loading signals: {str(e)}")
            return []

    def _load_historical(self) -> None:
        """Load historical signals from backup file."""
        path = "lae_historical_signals.json"
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                    self._update_metrics(data)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Error loading historical data: {str(e)}")

    def _hash(self, signal: Dict[str, Any]) -> str:
        """Generate a unique hash for a signal.
        
        Args:
            signal: Dictionary containing signal data.
            
        Returns:
            MD5 hash string of the signal.
        """
        return hashlib.md5(json.dumps(signal, sort_keys=True).encode()).hexdigest()

    def _calculate_rr_ratio(self, signal: Dict[str, Any]) -> Optional[float]:
        """Calculate risk-reward ratio for a signal.
        
        Args:
            signal: Dictionary containing signal data.
            
        Returns:
            Calculated risk-reward ratio or None if data incomplete.
        """
        tp = signal.get("take_profit", [])
        sl = signal.get("stop_loss")
        entry = signal.get("entry")
        
        if not all([tp, sl, entry]):
            return None

        avg_tp = sum(t["level"] * t.get("weight", 1) for t in tp) / sum(t.get("weight", 1) for t in tp)
        risk = abs(entry - sl)
        reward = abs(avg_tp - entry)
        return reward / risk if risk > 0 else 0

    def _analyze(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single trading signal.
        
        Args:
            signal: Dictionary containing signal data.
            
        Returns:
            Dictionary with analysis results.
        """
        result = {
            "id": self._hash(signal),
            "ts": signal.get("timestamp", "N/A"),
            "decision": signal.get("decision", "NO TRADE"),
            "confidence": signal.get("confidence", 0),
            "components": len(signal.get("components", {})),
            "success": False,
            "pnl": 0.0,
            "outcome": "not_evaluated",
            "rr": 0.0
        }

        if signal["decision"] == "NO TRADE":
            result["outcome"] = "no_signal"
            return result

        rr = self._calculate_rr_ratio(signal)
        if rr is None:
            result["outcome"] = "incomplete_data"
            return result

        result["rr"] = round(rr, 2)

        # Calculate success probability based on RR and confidence
        base_prob = 0.6 if rr >= 1.5 else 0.4
        boost = 0.3 * (signal["confidence"] / 100)
        final_prob = min(0.9, base_prob + boost)

        result["success"] = random.random() < final_prob
        result["outcome"] = "win" if result["success"] else "loss"
        
        # Calculate PnL
        tp = signal["take_profit"]
        sl = signal["stop_loss"]
        entry = signal["entry"]
        avg_tp = sum(t["level"] * t.get("weight", 1) for t in tp) / sum(t.get("weight", 1) for t in tp)
        risk = abs(entry - sl)
        reward = abs(avg_tp - entry)
        result["pnl"] = round(reward if result["success"] else -risk, 4)

        return result

    def _update_metrics(self, analyzed: List[Dict[str, Any]]) -> None:
        """Update metrics based on analyzed signals.
        
        Args:
            analyzed: List of analyzed signal dictionaries.
        """
        if not analyzed:
            return

        success = [s for s in analyzed if s["success"]]
        total = len(analyzed)
        success_count = len(success)
        
        self.metrics["total"] += total
        self.metrics["success"] += success_count
        self.metrics["rate"] = round((self.metrics["success"] / self.metrics["total"]) * 100, 2) if self.metrics["total"] > 0 else 0

        self.metrics["conf"].extend([s["confidence"] for s in analyzed])
        
        for s in analyzed:
            self.metrics["by_decision"][s["decision"]] += 1
            self.metrics["by_rr"][round(s["rr"], 1)].append(s["success"])
            
            if s["ts"] != "N/A":
                try:
                    hour = datetime.datetime.strptime(s["ts"], "%Y-%m-%d %H:%M:%S").hour
                    if s["success"]:
                        self.metrics["by_hour"][hour]["wins"] += 1
                    else:
                        self.metrics["by_hour"][hour]["losses"] += 1
                except ValueError:
                    self.logger.warning(f"Invalid timestamp format: {s['ts']}")

    def run(self, force: bool = False) -> Dict[str, Any]:
        """Run the analysis process.
        
        Args:
            force: If True, run analysis regardless of study window.
            
        Returns:
            Dictionary with analysis results and status.
        """
        if not (force or self._in_study_time()):
            self.logger.info("Outside study window hours")
            return {"status": "outside_study_time"}
        
        signals = self._load_signals()
        if not signals:
            self.logger.warning("No signals found to analyze")
            return {"status": "no_data"}

        # Analyze recent signals (max 50)
        to_study = signals[-50:] if len(signals) > 50 else signals
        analyzed = [self._analyze(s) for s in to_study]
        self.memory.extend(analyzed)
        self._update_metrics(analyzed)

        return {
            "status": "done",
            "analyzed": len(analyzed),
            "rate": self.metrics["rate"],
            "insights": self._insights(),
            "recommendations": self._recommend()
        }

    def _insights(self) -> List[str]:
        """Generate insights from analyzed data.
        
        Returns:
            List of insight strings.
        """
        if self.metrics["total"] < 10:
            return ["Not enough data for meaningful insights."]
        
        insights = []
        
        # Confidence analysis
        wins = [m["confidence"] for m in self.memory if m["success"]]
        losses = [m["confidence"] for m in self.memory if not m["success"]]
        
        if wins and losses:
            win_avg = statistics.mean(wins)
            loss_avg = statistics.mean(losses)
            if win_avg - loss_avg > 10:
                insights.append(f"🎯 High-confidence signals perform better ({win_avg:.2f}% vs {loss_avg:.2f}%)")
        
        # Time-based analysis
        if self.metrics["by_hour"]:
            best_hour = max(
                self.metrics["by_hour"].items(),
                key=lambda x: x[1]["wins"] / (x[1]["wins"] + x[1]["losses"]) if (x[1]["wins"] + x[1]["losses"]) > 0 else 0
            )
            win_rate = (best_hour[1]["wins"] / (best_hour[1]["wins"] + best_hour[1]["losses"])) * 100
            if win_rate > 60:
                insights.append(f"⏰ Hour {best_hour[0]}:00 has highest win rate ({win_rate:.1f}%)")
        
        # RR analysis
        if self.metrics["by_rr"]:
            best_rr = max(
                self.metrics["by_rr"].items(),
                key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0
            )
            if best_rr[1]:
                rr_win_rate = (sum(best_rr[1]) / len(best_rr[1])) * 100
                if rr_win_rate > 65:
                    insights.append(f"📊 RR {best_rr[0]} has win rate of {rr_win_rate:.1f}%")
        
        return insights if insights else ["No strong patterns identified yet."]

    def _recommend(self) -> List[str]:
        """Generate recommendations based on analysis.
        
        Returns:
            List of recommendation strings.
        """
        if self.metrics["total"] < 20:
            return ["Gather more signals for better strategy assessment."]
            
        recommendations = []
        
        if self.metrics["rate"] < 50:
            recommendations.append("🔧 Consider tuning signal logic - current win rate is low.")
            
        if len(self.metrics["conf"]) > 10:
            avg_conf = statistics.mean(self.metrics["conf"])
            if avg_conf < 60:
                recommendations.append("📈 Try increasing minimum confidence threshold.")
        
        # Check if certain decisions perform poorly
        if self.metrics["by_decision"]:
            worst_decision = min(
                self.metrics["by_decision"].items(),
                key=lambda x: x[1] / self.metrics["total"]
            )
            if worst_decision[1] / self.metrics["total"] > 0.3:
                recommendations.append(f"⚠ Review '{worst_decision[0]}' decisions - they account for {worst_decision[1]/self.metrics['total']:.1%} of signals")
        
        return recommendations if recommendations else ["✅ Strategy appears to be performing well."]

    def export(self) -> Dict[str, str]:
        """Export analysis results to JSON files.
        
        Returns:
            Dictionary with export status.
        """
        try:
            report_dir = Path("study_reports")
            report_dir.mkdir(exist_ok=True)
            
            with open(report_dir / "learning.json", "w") as f:
                json.dump(self.memory, f, indent=2)
                
            with open(report_dir / "metrics.json", "w") as f:
                json.dump(self.metrics, f, indent=2)
                
            self.logger.info("Successfully exported study reports")
            return {"status": "exported"}
        except (IOError, OSError) as e:
            self.logger.error(f"Failed to export reports: {str(e)}")
            return {"status": "export_failed", "error": str(e)}

if _name_ == "_main_":
    print("🔍 Starting Self-Study Engine")
    sse = SelfStudyEngine()
    result = sse.run(force=True)
    print("\n📊 Study Result:", json.dumps(result, indent=2))
    sse.export()
