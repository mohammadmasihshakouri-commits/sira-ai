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

from core.greeting_builder import build_greeting
from core.sira_runtime import handle_message
from core.channel_message import create_channel_message
from core.conversation_state import create_initial_state
from core.business_profile_loader import get_business_agent
from core.call_trace import (
    create_call_trace,
    add_turn,
    add_tool_call,
)

class SiraBrainProcessor(FrameProcessor):

    def __init__(self, business_profile_name, agent_key):
        super().__init__()

        self.business_agent = get_business_agent(
            business_profile_name,
            agent_key
        )

        self.conversation_state = create_initial_state()

        self.agent_config = self.business_agent["agent_config"]
        self.agent_identity = self.agent_config.get("agent_identity", {})
        self.call_settings = self.agent_config.get("call_settings", {})
        self.enabled_tools = self.agent_config.get("enabled_tools", [])
        self.playbooks = self.agent_config.get("playbooks", [])

        self.call_trace = create_call_trace(
             business_name=self.business_agent.get("business_name"),
             agent_key=self.business_agent.get("agent_key"),
)
        print("Loaded Business Agent:")
        print(f"Business: {self.business_agent.get('business_name')}")
        print(f"Agent Key: {self.business_agent.get('agent_key')}")
        print(f"Agent Type: {self.agent_config.get('agent_type')}")
        print(f"Agent Name: {self.agent_identity.get('agent_name')}")
        print(f"Enabled Tools: {self.enabled_tools}")
        print(f"Playbooks: {self.playbooks}")
        print(f"Call Settings: {self.call_settings}")

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            print(f"Customer: {frame.text}")

            channel_message = create_channel_message(
                channel="voice",
                user_id="local_test_user",
                message_text=frame.text,
                metadata={
                    "business": self.business_agent.get("business_name"),
                    "agent_key": self.business_agent.get("agent_key"),
                },
            )

            runtime_result = handle_message(
                channel_message=channel_message,
                conversation_state=self.conversation_state,
                enabled_playbooks=self.playbooks,
            )

            assistant_reply = runtime_result["reply"]
            source = runtime_result["source"]
            self.conversation_state = runtime_result["state"]
            tool_call = runtime_result.get("tool_call")

            print(f"Sira: {assistant_reply}")
            print(f"Source: {source}")
            print(f"State: {self.conversation_state}")

            self.call_trace = add_turn(
                trace=self.call_trace,
                customer_message=frame.text,
                assistant_reply=assistant_reply,
                source=source,
                state=self.conversation_state,
            )


            if tool_call:
                self.call_trace = add_tool_call(
                    trace=self.call_trace,
                    tool_name=tool_call["tool_name"],
                    input_data=tool_call["input"],
                    output_data=tool_call["output"],
                )

            await self.push_frame(
                TextFrame(assistant_reply),
                direction
            )

            return

        await self.push_frame(frame, direction)


async def main():

    print("Starting Sira Pipecat business-agent pipeline...")

    processor = SiraBrainProcessor(
        business_profile_name="cinematicket",
        agent_key="support"
    )

    greeting = build_greeting(processor.business_agent)

    pipeline = Pipeline([
        processor
    ])

    task = PipelineTask(pipeline)

    runner = PipelineRunner()

    if greeting:
     print(f"Sira Opening Greeting: {greeting}")

    await task.queue_frame(TextFrame("سلام وقتتون بخیر"))
    await task.queue_frame(TextFrame("من می‌خوام بلیتم رو کنسل کنم"))
    await task.queue_frame(TextFrame("کد رزرو من ۳۳۵۲۴۷۵۳ هست"))
    await task.queue_frame(TextFrame("بله درسته"))

    await task.queue_frame(EndFrame())

    await runner.run(task)
    
    print("\n=== CALL TRACE ===")
    print(processor.call_trace)
    print("Sira Pipecat business-agent pipeline finished")


if __name__ == "__main__":
    asyncio.run(main())