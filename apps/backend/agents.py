import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# --- SCHEMAS ---

class IntelSignal(BaseModel):
    headline: str
    impact_score: int = Field(..., ge=1, le=10)
    raw_content: str

class TradeAction(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"]
    confidence_score: float
    bull_thesis: str
    bear_antithesis: str
    execution_logic: str = Field(..., description="Steps for NSE/MCX execution")
    risk_buffer: str = Field(..., description="GST, Import Duty, or USD/INR impact")

# --- PROMPTS ---

BULL_PROMPT = """
You are the Artha Growth Strategist. Focus on maximizing alpha in the Indian market.
Global Signal: {signal}
Identify arbitrage between global benchmarks (e.g. COMEX Gold) and Indian markets (MCX), and macro tailwinds for NSE sectors.
"""

BEAR_PROMPT = """
You are the Artha Chief Risk Auditor with VETO power. 
Global Signal: {signal}
Bull's Thesis: {bull_thesis}
Critique the Bull's optimism. Factor in USD/INR slippage, Indian taxes (GST/STT), and liquidity risks. 
If risks outweigh gains, recommend HOLD or SELL.
"""

# --- ENGINE ---

class ArthaDecisionEngine:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=TradeAction)

    async def generate_action(self, signal: IntelSignal) -> TradeAction:
        # Step 1: Growth Thesis
        bull_resp = await (ChatPromptTemplate.from_template(BULL_PROMPT) | self.llm).ainvoke({"signal": signal.headline})
        
        # Step 2: Adversarial Audit
        bear_resp = await (ChatPromptTemplate.from_template(BEAR_PROMPT) | self.llm).ainvoke({
            "signal": signal.headline,
            "bull_thesis": bull_resp.content
        })

        # Step 3: Synthesis
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", "Synthesize the debate into a structured TradeAction JSON."),
            ("human", "Signal: {sig}\nBull: {bull}\nBear: {bear}\n{format_instructions}")
        ])
        
        chain = synthesis_prompt | self.llm | self.parser
        return await chain.ainvoke({
            "sig": signal.model_dump_json(),
            "bull": bull_resp.content,
            "bear": bear_resp.content,
            "format_instructions": self.parser.get_format_instructions()
        })