import asyncio
import sys
from pathlib import Path

from pipecat.frames.frames import TextFrame, EndFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app import process_text


class SiraBrainProcessor(FrameProcessor):

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            print(f"Customer: {frame.text}")

            assistant_reply, source = process_text(frame.text)

            print(f"Sira: {assistant_reply}")
            print(f"Source: {source}")

            await self.push_frame(
                TextFrame(assistant_reply),
                direction
            )

            return

        await self.push_frame(frame, direction)


async def main():

    print("Starting Sira Pipecat brain pipeline...")

    processor = SiraBrainProcessor()

    pipeline = Pipeline([
        processor
    ])

    task = PipelineTask(pipeline)

    runner = PipelineRunner()

    await task.queue_frame(TextFrame("سلام، من برای پشتیبانی تماس گرفتم"))
    await task.queue_frame(EndFrame())

    await runner.run(task)

    print("Sira Pipecat brain pipeline finished")


if __name__ == "__main__":
    asyncio.run(main())
