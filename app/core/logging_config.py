import logging
import json
from datetime import datetime, timezone

class StructuredFormatter(logging.Formatter):
    STANDARD_FIELDS = {
    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
    'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
    'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
    'thread', 'threadName', 'processName', 'process', 'message',
    'taskName'
    }

    def format(self, record):
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level" : record.levelname,
            "message" : record.getMessage(),
            "module" : record.module,
            "function": record.funcName,
            
        }
            
        for key, value in record.__dict__.items():
            if key not in self.STANDARD_FIELDS and key not in log_data:
                log_data[key] = value

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)
        

def setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []
    root_logger.addHandler(handler)     


