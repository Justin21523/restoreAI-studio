import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    PROGRESS = "progress"
    STARTED = "started"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class EventManager:
    """Manage server-sent events for job progress"""

    def __init__(self):
        self.connections: Dict[str, List[asyncio.Queue]] = {}

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        """Subscribe to job events"""
        if job_id not in self.connections:
            self.connections[job_id] = []

        queue = asyncio.Queue()
        self.connections[job_id].append(queue)
        logger.debug(f"New subscriber for job {job_id}")
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue):
        """Unsubscribe from job events"""
        if job_id in self.connections:
            if queue in self.connections[job_id]:
                self.connections[job_id].remove(queue)
            if not self.connections[job_id]:
                del self.connections[job_id]

    async def publish(self, job_id: str, event_type: EventType, data: Dict[str, Any]):
        """Publish event to all subscribers"""
        if job_id not in self.connections:
            return

        event = {"type": event_type.value, "data": data, "timestamp": time.time()}

        message = f"data: {json.dumps(event)}\n\n"

        # Send to all subscribers
        for queue in self.connections[job_id][:]:
            try:
                await queue.put(message)
            except Exception as e:
                logger.error(f"Failed to send event to subscriber: {e}")
                # Remove faulty subscriber
                self.connections[job_id].remove(queue)

    async def generate_events(self, job_id: str):
        """Generate SSE stream for job"""
        queue = await self.subscribe(job_id)
        try:
            while True:
                message = await queue.get()
                yield message
                queue.task_done()
        finally:
            self.unsubscribe(job_id, queue)


# Global event manager instance
event_manager = EventManager()
