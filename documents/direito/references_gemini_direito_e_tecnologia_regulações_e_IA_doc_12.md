# AI Compliance: The Global

Guide to International

AI Regulations

AI compliance is the practice of proving your AI systems meet legal, regulatory, and standards-based obligations across every jurisdiction where you develop, deploy, or use them. This guide covers regulations with enforcement mechanisms and penalties, updated monthly.

The EU AI Act is enforcing now. US states are splintering. International standards are the lingua franca. No single baseline works everywhere, and that's why a unified framework matters.

## What Is AI Compliance? The State of Global AI Regulations in 2026

AI compliance is the practice of proving your AI systems meet legal, regulatory, and standards-based obligations across every jurisdiction where you develop, deploy, or use them. As global AI regulations multiply and diverge, compliance means navigating jurisdiction-specific, sector-specific, and risk-specific requirements: before deployment, during operation, and after incidents occur.

Compliance obligations bind **providers** (building AI), **deployers** (using AI operationally), **importers** (distributing AI into regulated markets), and **resellers** (integrating or bundling AI). Under the EU AI Act, liability cascades. A cloud provider fine-tuning customer data can trigger high-risk classification, making the deployer responsible too. You can't isolate your way out.

**Compliance, governance, and risk management are entangled but distinct.** Governance is structure: who decides, how conflicts escalate, what policies bind the org. Risk management is process: identify harms, measure likelihood and severity, implement controls. Compliance is proof: regulators and auditors verify you met the specific obligations they've written down. You can have bulletproof governance and still fail compliance if you missed a jurisdiction's narrow legal requirement. See AI Governance and AI Risk Management for those distinct pillars.

No single baseline works everywhere. EU: risk-tiered under the AI Act. Texas: governance transparency under TRAIGA. California: frontier model reporting. Japan: voluntary adherence. That's why a unified framework matters: one control architecture that maps to every regulation simultaneously.

**The ground truth in 2026:**

-
**EU is enforcing now.**The AI Act took effect August 1, 2024. Prohibited practices became enforceable February 2, 2025. General-purpose AI obligations took effect August 2, 2025. High-risk systems have until August 2, 2026. That's your real deadline. If you're in scope, you're already under the world's strictest regime. -
**US is splintered.**No federal AI law. Instead: 40+ states with active legislation. Texas TRAIGA enforces as of January 1, 2026. California SB 53 live January 1, 2026. Colorado SB 24-205 delayed to June 30, 2026 with uncertain future. The December 2025 federal preemption order created legal chaos. DOJ is challenging state laws, but compliance risk remains concrete and immediate. Agencies (FTC, FDA, FINRA) are already acting on algorithmic bias under existing authority. No coherent baseline. You navigate by jurisdiction. -
**International standards are the lingua franca.**ISO 42001 (published December 2023), NIST AI RMF (January 2023), and OWASP Top 10 for LLMs (2025) now define enterprise governance. Not legally mandatory, but Fortune 500 procurement teams demand them. Insurance underwriters require them. They're the common currency across jurisdictions.

This guide covers regulations with enforcement mechanisms and penalties. Not aspirational. Teeth.

→ Explore the Modulos AI Governance Platform to operationalize compliance at scale.

## Why AI Compliance Matters in 2026

The largest penalty isn't a fine. It's a scandal.

Amazon's recruiting tool discriminated against women. Scrapped. Textbook case in AI bias. A deepfake cost a multinational millions in fraud. Shattered trust in voice auth. For Fortune 500, a Wall Street Journal investigation about your AI's unfairness tanks brand, stock price, customer pipeline overnight. A €35 million EU AI Act fine can be litigated for years. A scandal cannot.

**Procurement is demanding proof.** Fortune 500 RFPs now require ISO 42001 certification, SOC 2 with AI-specific controls, or third-party bias audits before contract award. Enterprise customers treat non-compliant vendors as uninsurable liability. You want to sell? Prove compliance.

**Insurance forced the issue.** Cyber and professional liability carriers are excluding or conditioning coverage for unaudited AI systems. Underwriters ask: Do you have an AI governance policy? Running bias tests? Done an algorithm impact assessment? Without documented compliance, you're uninsurable, or paying multiples of premium.

**Penalties are now concrete.** EU AI Act: 7% of annual global turnover (or €35 million, whichever is higher) for serious violations. Colorado UDAP: consumer protection penalties. Texas TRAIGA: $100,000 per violation, but 60-day cure period. California SB 53: up to $1 million per violation for frontier AI companies failing to report. These are floors. Reputational damage is the ceiling.

## The Global AI Regulatory Map

This guide covers regulations with enforcement mechanisms. Countries actively developing frameworks not included here.

| Jurisdiction | Regulation | Status | Effective Date | Key Obligation | Penalty Ceiling |
|---|---|---|---|---|---|
EU | AI Act | Enforcing (phased) | Feb 2025 (banned); Aug 2025 (GPAI); Aug 2026 (general) | Risk-tier compliance, impact assessments | 7% global turnover / €35M |
US Federal | Trump EO + AI Action Plan | Enacted | Dec 2025 / July 2025 | Preemption + sector guidance | N/A (executive action) |
Colorado | SB 24-205 | In amendment | 30 June 2026 (in flux) | Algorithmic discrimination prevention | UDAP penalties |
Texas | TRAIGA (HB 149) | Enforcing | 1 Jan 2026 | AI governance + transparency | $100K/violation |
California | SB 53 | Enforcing | 1 Jan 2026 | Frontier AI transparency | Up to $1M/violation |
NYC | Local Law 144 | Enforcing | 1 Jan 2023 | Bias audits for hiring AI | Civil penalties |
UK | Sector-based | Pending legislation | TBD | Sector regulator guidance | Varies by sector |
China | Layered regulations | Enforcing | Aug 2023 onwards | Filing, labeling, content | Administrative |
Saudi Arabia | PDPL + AI Framework | Enforcing | Sept 2024 / 2026 | Data protection + AI governance | SDAIA enforcement |
UAE | PDPL | Enforcing (transition) | Jan 2026 | Data protection + AI | Regulatory penalties |
Council of Europe | Framework Convention | In force | 1 Nov 2025 | Human rights + democracy | International accountability |
Vietnam | Decree 13/2023 + Law 134/2025 | In force | July 1, 2023 / March 1, 2026 | Data protection for AI; AI-specific governance | Administrative |
South Korea | Framework Act on AI | In force | 22 Jan 2026 | High-impact AI obligations | Administrative |
Japan | AI Promotion Act | In force | June 2025 | Non-punitive governance | None (soft law) |
India | DPDP + deepfake rules | In force (phased) | Nov 2025 / Feb 2026 | Data protection + synthetic content | Up to ₹250 crore |
Brazil | PL 2338 | Stalled | TBD | Risk-based governance | TBD |
Canada | AIDA (C-27) | Withdrawn | N/A | N/A | N/A |
ISO 42001 | AI Management System | Published | Dec 2023 | Management system + certification | N/A (voluntary) |
NIST AI RMF | Risk Management Framework | Published | Jan 2023 | Govern / Map / Measure / Manage | N/A (voluntary) |
OWASP Top 10 LLM | LLM Security Guidelines | Published | 2025 | LLM vulnerability prevention | N/A (voluntary) |

## EU AI Act: The World's First Comprehensive AI Law

The EU AI Act (Regulation 2024/1689) is the world's first binding, comprehensive AI regulation, and it entered into force on August 1, 2024.

This is not aspirational guidance. It is law. It applies to any organization placing AI systems on the EU market or deploying AI that affects EU residents, regardless of where your company is incorporated. The regulation has extraterritorial reach, and the penalties are severe enough that even non-European firms must take it seriously.

### Overview and Risk Tiers

The EU AI Act operates on a risk-based architecture with four tiers, each imposing escalating obligations.

**Unacceptable-risk AI is banned outright.** This includes social scoring systems, untargeted facial recognition scraping, emotion recognition in workplaces and schools, and AI that exploits vulnerable groups through manipulation. If your system falls into this category, you cannot deploy it in the EU. For the full list, see EU AI Act prohibited practices.

**High-risk AI** (Annex III) requires mandatory conformity assessment, risk management, data governance, human oversight, transparency labeling, and accuracy testing. High-risk categories include employment decisions, education and vocational training, credit scoring, criminal justice and law enforcement, border management, and critical infrastructure. If you deploy a high-risk system, you must conduct a formal conformity assessment, maintain detailed technical documentation, implement continuous monitoring, and achieve CE marking before placing the system on the market.

**Limited-risk AI** requires transparency: chatbots must disclose that users are interacting with AI; deepfakes and synthetic media must be labeled.

**Minimal-risk AI** faces no legal obligations, though the regulation encourages voluntary compliance codes.

For deeper context on scoping and risk classification, see EU AI Act deep dive and EU AI Act scoping guide.

### Timeline and Enforcement Status

The phased schedule matters. These dates are your roadmap.

**February 2, 2025.** Prohibited practices became enforceable. If you're using banned AI (social scoring, untargeted facial recognition scraping, emotion recognition in hiring), you are already in violation.

**August 2, 2025.** GPAI provider obligations effective. Any org offering foundational or large language models to the EU market must comply: technical documentation, training data summaries, copyright compliance disclosures. Now.

**August 2, 2026.** The main enforcement date. High-risk systems must be fully compliant. Conformity assessments completed. CE marking in place. This is the date your compliance team works backward from.

**The Digital Omnibus uncertainty.** The European Commission proposed amendments in November 2025 to simplify implementation. Trilogue negotiations ongoing (as of April 2026), expected conclusion May 2026. The proposal may delay standalone high-risk system deadlines to December 2, 2027, and embedded systems to August 2, 2028. The core framework (risk tiers, prohibited practices, penalties) stays fixed. But timelines may shift. Prudent approach: assume August 2026 is firm, treat extensions as windfall.

The EU AI Office is operational. Member states are designating supervisory authorities. Enforcement accelerates second half 2026. See EU AI Act high-risk compliance deadline 2026 for full timeline breakdown.

### The Digital Omnibus: What's Changing

The Digital Omnibus is not a repeal or overhaul; it's a streamlining package.

Key proposed changes under trilogue negotiation (April 2026):

- •Possible delay of high-risk implementation deadlines (Council and Parliament aligning on December 2027 for standalone, August 2028 for embedded systems)
- •Simplification of technical documentation requirements
- •Narrowing of some Annex III categories (particularly employment and education AI)
- •More flexibility in compliance timelines for organizations already in good-faith implementation

What's not changing: the prohibited practices list is fixed, the penalty structure is fixed, the risk-based framework is fixed. This is a refinement, not a retooling. For the latest on Omnibus negotiations, see the Modulos EU AI Act page, which we update as trilogue progresses.

### GPAI Obligations

General-purpose AI (GPAI) providers face a separate compliance track from high-risk system deployers.

GPAI transparency obligations became effective August 2, 2025. Providers must supply technical documentation on model capabilities and limitations, copyright compliance information, training data summaries, and content provenance mechanisms. Commission enforcement begins August 2, 2026.

GPAI models with systemic risk (those exceeding 10²⁵ FLOPs during training or designated by the Commission) face additional adversarial testing, incident reporting, and cybersecurity requirements.

A Code of Practice for GPAI was finalized July 10, 2025, providing practical guidance on transparency and mitigation measures.

### Penalties and Conformity Assessment

The penalty structure is designed to hurt.

**Prohibited practices violations:**up to €35 million or 7% of global annual turnover (whichever is higher)**Other violations**(high-risk non-compliance, governance failures): up to €15 million or 3% of turnover**Providing false information**to authorities: up to €7.5 million or 1% of turnover

Conformity assessment is the mechanism for proving compliance. For most high-risk systems, you conduct a self-assessment. For biometric systems, a third-party notified body must be involved. Once assessed, CE marking is required: your formal attestation that the system complies.

For technical guidance, see Conformity Assessment and CE Marking and EU AI Act scoping. For a comprehensive walkthrough, download our EU AI Act guide.

## US Federal AI Policy: From Biden to Trump

The US has no federal AI statute. The landscape is a narrative of executive action, agency guidance, and a fundamental shift in regulatory philosophy between administrations.

### Biden's EO 14110: The Historical Baseline

On October 30, 2023, President Biden signed Executive Order 14110, "Safe, Secure, and Trustworthy AI." It was the most comprehensive federal AI action at the time. It required safety testing for powerful models, red-teaming, watermarking, and agency guidance on AI use within government.

On January 20, 2025, Trump revoked EO 14110 on his first day in office. The NIST AI Risk Management Framework and related technical standards survived the revocation and continue to be refined. Some federal agencies continued AI governance work independently.

### Trump's AI Executive Orders and the AI Action Plan

President Trump's approach centers on competitiveness and deregulation.

In January 2025, Trump signed an AI executive order emphasizing "removing barriers to American AI innovation." In July 2025, OSTP released "America's AI Action Plan," incorporating over 2,400 public comments. The strategy focuses on compute infrastructure, export controls, and talent, not compliance obligations. Voluntary industry commitments are preferred over regulatory mandates.

### The December 2025 Preemption Executive Order

On December 11, 2025, Trump signed "Ensuring a National Policy Framework for AI," the order that rewrote the compliance landscape.

The core mechanism is federal preemption of state AI laws:

**DOJ Litigation Task Force**(30-day mandate): challenge state AI laws as unconstitutional or burdens on interstate commerce**FCC Rulemaking**(90 days): establish federal AI model reporting standards to preempt state requirements**FTC Policy Statement**(90 days): clarify FTC Act application to AI

Carve-outs exist for child safety, AI compute infrastructure, and state government procurement.

**Practical impact:** This creates genuine uncertainty. State laws discussed in the next section may be preempted, delayed, or narrowed by federal action. The safest approach: comply with enacted state laws while monitoring preemption litigation. Non-compliance carries concrete risk; preemption is speculative.

### Sector Regulators Filling the Gap

With no federal statute, sector regulators act independently. The **FTC** pursues enforcement actions on deceptive AI claims and algorithmic bias. **SEC and FINRA** focus on AI governance in financial services; FINRA's 2026 report identifies agentic AI supervision as the emerging focal point. The **FDA** has authorized over 1,250 AI-enabled medical devices as of July 2025 and is moving toward formal premarket approval for standalone AI medical devices.

## The US State AI Law Patchwork

The IAPP tracks 40+ states with active AI legislation. Without federal law, states are the regulatory surface. This section covers the laws with cross-industry teeth, and the December 2025 preemption order adds legal uncertainty to all of it.

### Colorado AI Act Compliance (SB 24-205)

Most closely watched state AI law in the US. Future uncertain as of April 2026.

Signed May 17, 2024, originally effective February 1, 2026. Amended August 28, 2025 (SB 25B-004): delayed to June 30, 2026 for further refinement. As of April 2026, additional amendments or repeal under discussion. DOJ preemption task force specifically targeted Colorado.

Core obligation: high-risk AI systems must use "reasonable care" against algorithmic discrimination. High-risk = materially impacts employment, credit, housing, insurance, healthcare, education. AG enforcement, UDAP penalties.

**Reality check:** Don't assume June 2026 is firm. Monitor Colorado legislature and DOJ litigation simultaneously. Prudent orgs prepare for both scenarios: compliance and preemption.

### Texas TRAIGA Compliance (HB 149)

In force as of January 1, 2026. Signed June 22, 2025.

Requires reasonable care, transparency, testing, impact assessments. Unique feature: AG enforcement includes mandatory 60-day cure period before penalties. That's remediation time, a material difference from other states.

Penalties: up to $100,000 per violation. See TRAIGA compliance guide for implementation.

### California: From SB 1047 to SB 53

SB 1047 was ambitious (safety testing, kill-switch mandates for frontier models). Vetoed by Newsom September 2024 as overreach. Successor, SB 53 (Frontier AI Transparency Act), signed September 29, 2025, effective January 1, 2026.

SB 53 requires frontier AI developers to disclose capabilities, limitations, testing results, safety measures. Penalties: up to $1 million per violation, AG enforced. AB 2013 (training data transparency) also in effect, complementary with narrower scope.

### NYC Local Law 144

In force since January 1, 2023. One of the oldest AI-specific laws in the US.

Requires annual independent bias audits, public results posting, candidate notification. December 2025 Comptroller audit found enforcement gaps: 75% of complaints misrouted, 1 violation detected in 32 companies surveyed while auditors identified 17+ potential violations. Enforcement tightening in 2026.

### Utah AI Policy Act

Effective May 1, 2024. Requires disclosure when consumers interact with AI in regulated transactions (financial, employment, insurance, housing). Narrower than Colorado or Texas, but enforced.

### Tennessee ELVIS Act

Effective July 1, 2024. Protects voice and likeness from unauthorized AI replication. First US law explicitly addressing AI-generated voice/likeness cloning. Statutory damages: up to $12,000 per violation, private right of action.

### The Broader State Landscape

Most states have some AI legislative activity: bills, executive orders, task forces, commissions. The IAPP AI legislation tracker is definitive. Notable: Illinois Video Interview Act, Connecticut insurance AI rules, Massachusetts and Washington active bills.

Three patterns dominate. First, employment AI (hiring and termination decisions). Second, consumer protection (algorithmic discrimination and deceptive claims). Third, deepfakes and voice cloning (unauthorized synthetic media). Don't map compliance to all 40+ states. Identify which states you operate in, build controls for those three vectors.

### The Preemption Question

Does the December 2025 federal preemption executive order invalidate state AI laws?

Not yet. Executive orders don't directly override state law. The DOJ litigation task force must bring and win challenges. The Supreme Court has historically deferred to state law in health, safety, and consumer protection. Litigation could take years.

**Practical reality:** Comply with enacted state laws. Monitor preemption litigation. Non-compliance risk is concrete and immediate. Preemption risk is speculative and years away. Don't use preemption uncertainty as an excuse to skip compliance. Comply now, adjust if preemption succeeds.

## UK: Pro-Innovation, Sector-Based Approach

The UK has no comprehensive AI law and is unlikely to have one before late 2026 at earliest.

The Labour government (took office July 5, 2024) committed to "appropriate legislation" for powerful AI but explicitly prioritizes economic growth. The *AI Opportunities Action Plan* (published January 13, 2025) set the tone: 50 recommendations focused on competitiveness, not compliance.

The *AI Regulation Bill* was reintroduced to the House of Lords (first reading March 4, 2025) as a **Private Members' Bill** without government backing. It remains in committee with no clear timeline and low passage probability. Until comprehensive legislation arrives, UK compliance is sector-by-sector:

**ICO**applies data protection law (UK GDPR, Data Protection Act 2018) to AI systems processing personal data**FCA**regulates algorithmic decision-making in financial services**CMA**scrutinizes AI-driven market concentration**Ofcom**oversees AI in online platforms and broadcast media

**For compliance teams:** Treat the UK as collection of sectoral mandates. If your AI system processes UK resident data, GDPR compliance is mandatory. If it makes decisions in finance, FCA standards apply. No overarching UK AI compliance obligation equivalent to the EU Act exists.

## China's Layered AI Regulation

China was first to enforce binding generative AI regulations (August 2023), layering multiple targeted rules rather than one comprehensive law.

The *Interim Measures for Generative AI Services* (effective August 15, 2023), administered by CAC and six co-regulators, require AI-generated content align with "socialist core values," mandate user identity verification, enforce data security. Despite the name, "interim" has become permanent: 2+ years on, no replacement announced.

Layered atop this:

**Deep Synthesis Provisions**(January 10, 2023): govern deepfakes and synthetic media**Algorithm Recommendation Provisions**(March 1, 2022): transparency for content recommendation systems**AIGC Labeling Measures**(March 2025; national standards September 1, 2025): require explicit and implicit marking of AI-generated content, including a watermarking mandate without parallel elsewhere**Amended Cybersecurity Law**(effective January 1, 2026): introduces dedicated AI compliance provisions

Enforcement is administrative and opaque. No public case law database. Fines and service suspensions issued by CAC without published reasoning.

**For compliance teams:** If you serve Chinese users or operate in China, compliance is mandatory and non-negotiable. The layered approach requires mapping obligations across four to five overlapping statutes. Budget for ongoing CAC monitoring; rules evolve faster than formal amendments.

## Middle East: Saudi Arabia, UAE, and the Gulf

The GCC states integrate AI governance into data protection frameworks rather than creating standalone AI laws. And enforcement is real.

### Saudi Arabia

The *Personal Data Protection Law* (PDPL), administered by SDAIA, took effect September 14, 2024. SDAIA has issued 48 violation decisions in 2024 and 2025. This is not paper regulation.

The government declared 2026 the "Year of AI" and released an *AI Adoption Framework*, mandatory for the public sector. Five pillars: data governance, model accountability, transparency, human oversight, risk management.

### United Arab Emirates

The PDPL takes effect January 1, 2026 (transition period); full compliance January 1, 2027. The CBUAE issued AI/ML guidance (February 2026) applying to all licensed financial institutions: board accountability, model inventory, annual bias testing, kill-switch requirements, consumer opt-out, and human review rights. AI governance is integrated within data protection, with no standalone AI statute.

### Qatar, Bahrain, and Oman

Qatar's QCB AI Guidelines have been mandatory since September 2024, with a draft AI law under consideration and QFMA capital markets rules in place. Bahrain's Shura Council approved a 38-article AI Regulation Law in April 2024 (under review); a General AI Policy has been active since May 2025. Oman's National AI Policy (in force April 2025) covers governance standards, assessments, documentation, and compliance reporting.

For the full GCC regulatory breakdown: Middle East AI Regulations Guide

## Council of Europe Framework Convention: The First Binding International AI Treaty

The Council of Europe Framework Convention on AI (CETS 225) is the world's first legally binding international treaty on artificial intelligence, in force since November 1, 2025.

Signed by 37+ countries (September 2024), including EU member states, US, UK, Canada, Japan, Australia. The EU gave formal consent in early 2026; the US and UK have ratified. The Convention establishes binding obligations on AI development and deployment framed around human rights, democracy, rule of law.

It does not prescribe product-level controls like the EU AI Act. Instead, it sets a human-rights baseline. China and Russia are not signatories, which limits practical global reach.

**For compliance teams:** The Convention doesn't impose direct conformity assessment obligations. But it creates an international accountability floor beneath jurisdiction-specific rules. If you operate across treaty signatories, the Convention's framing may influence how courts and regulators interpret your obligations.

## Asia-Pacific & Latin America: At a Glance

Outside the EU, US, China, and Gulf, AI regulation ranges from enacted-but-soft (Japan) to binding-and-strict (Vietnam, South Korea) to stalled (Brazil, Canada).

| Jurisdiction | Law / Framework | Status | Effective | Binding? | Key Feature |
|---|---|---|---|---|---|
Vietnam | Decree 13/2023/ND-CP | Enacted | July 1, 2023 | Yes | Data protection decree (not AI-specific); strict requirements on AI processing Vietnamese citizen data |
Vietnam | Law 134/2025 | Enacted | March 1, 2026 | Yes | Standalone AI law; one of APAC's strictest; covers high-risk AI, transparency, data governance |
South Korea | Framework Act on AI | Enacted | 22 Jan 2026 | Yes | Second country after EU with comprehensive AI legislation; covers high-impact AI |
Japan | AI Promotion Act | Enacted (phased) | June 2025 / Sept 2025 | No | No penalties; soft law; AI Basic Plan adopted Dec 2025 |
India | DPDP Act + deepfake rules | Enacted (phased) | Nov 2025 / Feb 2026 | Yes | No standalone AI Act; world's first deepfakes governance framework |
Singapore | Model AI Governance Framework | Voluntary | Ongoing | No | Practical, adoption-oriented; global reference model |
Australia | Voluntary AI Safety Standard | Voluntary | Oct 2025 | No | Mandatory guardrails postponed; sector regulators lead |
Brazil | PL 2338/2023 | Stalled | N/A | N/A | Senate passed Dec 2024; Chamber committee formed Apr 2025; no action date |
Canada | AIDA (C-27) | Withdrawn | N/A | N/A | Died on order paper Jan 2025; no replacement tabled |

### Vietnam AI Law and Data Protection

Vietnam represents the sharpest regulatory acceleration in APAC. Decree 13/2023 (effective July 1, 2023) predates the EU AI Act and imposes strict requirements on AI processing Vietnamese citizen data, with particularly tight cross-border data flow restrictions.

Law 134/2025 (effective March 1, 2026) is a standalone binding AI statute, one of the strictest globally. It covers high-risk AI systems, transparency obligations, and data governance requirements. Mandatory for any enterprise with Vietnamese user exposure.

### South Korea AI Act (Framework Act on AI)

South Korea became the second country after the EU to enact comprehensive AI legislation. The Framework Act on AI (effective January 22, 2026) mirrors EU high-risk categories, signaling the EU model is becoming the global template. Covers high-impact AI systems with obligations for risk assessment, transparency, and human oversight.

### Japan, India, Brazil, and Canada

**Japan** took a different path. The AI Promotion Act is intentionally non-binding, with no enforcement mechanism and governance framed as industry self-regulation. Low compliance cost, but watch for sector regulators layering rules atop the framework.

**India** is critical. With 1.4 billion people and rapid AI adoption, India has no standalone AI Act. AI is regulated under the DPDP Act (phased through May 2027) and sector rules. The February 2026 deepfakes framework is a global precedent.

**Brazil** is stalled, not abandoned. PL 2338 passed the Senate December 10, 2024; a Chamber special committee was formed April 2025 but momentum is uncertain. **Canada's AIDA** died with Parliament on January 6, 2025. No replacement as of April 2026. Both countries rely on existing data protection laws (LGPD, PIPEDA) for AI governance.

## The Standards That Help You Comply With All of Them

Regulations diverge by jurisdiction but standards converge. Standards are how you build one framework that maps to every regulation simultaneously.

Every regulator asks the same underlying questions: What AI systems do you operate? What are their risks? What controls do you have? Can you prove it? Standards provide the common language to answer all of them at once.

### ISO/IEC 42001: AI Management System

ISO/IEC 42001 is the only certifiable management system standard built specifically for AI governance.

Published December 2023, it codifies organizational infrastructure: governance structure, risk management, compliance, continual improvement. Clauses 4 through 10 define the system; Annexes A through D map controls to lifecycle phases.

Certification path: gap analysis, then implementation, then internal audit, then Stage 1/Stage 2 external audit. Six to nine months for a mid-size company.

**The key distinction:** ISO 42001 certifies your management system exists, not that your AI is compliant. It's a governance credential. Regulators view it as evidence you've institutionalized AI oversight. It's not a compliance guarantee. For that analysis, see Why certification alone isn't enough.

For multinationals, ISO 42001 maps directly to regulatory requirements. The EU AI Act's conformity assessment maps to 42001's control objectives. NIST AI RMF's governance function maps to 42001's leadership clauses. One implementation satisfies multiple regulators.

Resources: ISO/IEC 42001 overview · Clauses 4–10 · Annexes A–D · Pioneering AI management systems

### ISO/IEC 23894: AI Risk Management

Published February 2023. ISO 42001 asks "do you have a system?" ISO 23894 asks "how rigorously do you assess AI-specific risks?"

Extends ISO 31000 (general risk management) for AI characteristics: opacity, autonomy, data dependency. Covers model risk, data risk, integration risk, systemic risk. Non-certifiable but standard practice as the risk foundation for ISO 42001 implementation.

### ISO/IEC 42005: AI Impact Assessment

Published May 2025, the newest addition. Focuses on societal, human, organizational impacts (not just technical risk). Built to support regulations requiring impact assessment. Integrates with ISO 42001.

### NIST AI Risk Management Framework (RMF) + GenAI Profile

The de facto standard for AI risk management in the US, and increasingly elsewhere.

NIST AI RMF 1.0 (published January 26, 2023) organizes governance into four functions: **Govern** (establish roles, policies, oversight) → **Map** (identify systems, characterize context) → **Measure** (evaluate performance, safety, fairness) → **Manage** (implement controls, incident response, improve). Built by 2,500+ participants.

The GenAI Profile (NIST.AI.600-1, July 26, 2024) adds 200+ actions for LLM-specific risks: prompt injection, hallucinations, training data integrity, sensitive information leakage.

**NIST and ISO are complementary.** NIST provides methodology; ISO provides structure. Most enterprises use both.

### OWASP Top 10 for LLMs + Agentic AI

The security counterpart to governance frameworks, and increasingly mandatory in RFPs.

The 2025 edition ranks: Prompt Injection, Sensitive Information Disclosure, Supply Chain Vulnerabilities, Data & Model Poisoning, Improper Output Handling, Excessive Agency, System Prompt Leakage (new), Vector/Embedding Weaknesses (new), Misinformation, Unbounded Consumption.

OWASP Top 10 for Agentic AI addresses autonomous agents accessing databases, APIs, or business systems unsupervised.

Details: OWASP Top 10 for LLMs · OWASP Agentic AI

### OECD AI Principles + G7 Hiroshima Code of Conduct

Soft-law baseline adopted by 46 countries. Five principles: transparency, accountability, robustness, privacy, human oversight.

The G7 Hiroshima Code of Conduct (agreed October 30, 2023; endorsed by G7 leaders December 2023) applies to frontier AI developers. The OECD Reporting Framework launched February 2025; 19 companies reported by April 2025. Non-binding, but increasingly the benchmark regulators measure against.

### Integrating ISO 27001 + ISO 42001

Most enterprises already have ISO 27001 (information security). Both follow Annex SL structure, enabling a shared framework for integrated audits, unified policy, single management review.

Integrated implementation means: one audit covering both information security and AI governance. One policy framework. One training program. For organizations starting from zero: ISO 27001 first (foundation), then ISO 42001, then integrate.

Resources: ISO 27001 + 42001 integration · AI governance taxonomy · ISO 27001 Annex A

## Sector-Specific AI Rules: Healthcare, Finance, Employment, Platforms

Beyond horizontal AI laws, sector regulators are imposing AI-specific obligations. In some cases, they're ahead of the general legislation.

### Healthcare: FDA AI/ML Governance

The FDA's AI/ML draft guidance (January 6, 2025) establishes a Total Product Life Cycle approach for AI-enabled medical devices. Over 1,250 AI devices authorized as of July 2025. Key expectations: pre-market validation, Predetermined Change Control Plans (PCCPs) for post-launch updates, real-world performance monitoring, and training data transparency.

### Financial Services: SEC, FINRA, and Agentic AI

FINRA's 2026 report identifies agentic AI supervision as the emerging focal point: autonomous systems executing trades or decisions with limited human oversight. SEC enforcement focuses on deceptive AI claims; no AI-specific rule yet. FINRA Rule 3110 (Supervision) applies to GenAI tools, requiring written procedures for AI-based recommendations.

### Employment

NYC LL 144 is the model. Illinois AI Video Interview Act adds state-level echoes. Momentum across states on hiring AI transparency and bias auditing.

### Platforms: EU DSA

The Digital Services Act intersects with the AI Act on content moderation and recommendation systems. VLOPs face annual independent audits and transparency reports (first harmonized reports due early 2026). Penalties up to 6% of global turnover.

## How to Build a Global AI Compliance Program

Rules are diverging by jurisdiction at an accelerating rate, but the underlying requirements converge on the same core.

Here's the structural truth: the complexity is in the *mapping*, not in the *doing*. Every regulator asks identical underlying questions. What AI systems do you operate? What are their risks? What controls do you have? Can you prove it? The answers differ by jurisdiction only in specifics, not in substance. EU requires conformity assessment; Colorado requires risk notification; China requires algorithm registration. The underlying control infrastructure is the same.

Good governance (one unified control framework mapped to every applicable regulation) is the only scalable answer. Without it, you hire a compliance team per jurisdiction. With it, you hire one team that speaks every regulatory dialect.

### The Six-Step Framework

**1. Inventory.** What AI systems do you have? Where are they deployed? Who are the users and affected persons? What jurisdictions apply? Inventory is ongoing. Every quarter, ask what's new and what's retired.

**2. Classify.** Map each system to applicable regulations and risk tiers. A hiring AI deployed in the EU and US might be EU AI Act high-risk, Colorado "consequential AI," NYC LL 144-scoped, GDPR DPIA-required, and NIST RMF-applicable, all simultaneously.

**3. Control.** Implement one set of core controls (ISO 42001 + NIST RMF as backbone) and map them to each regulation's specific requirements. The EU AI Act requires "technical documentation": your ISO 42001 governance produces it. Colorado requires "risk notification": your NIST risk register feeds it. Controls span technical (bias testing, adversarial testing, explainability), organizational (governance structure, roles, training), and process levels (risk assessment, change management).

**4. Evidence.** Compliance is about evidence. Document everything: governance records, risk assessments, control testing results, monitoring data, improvement actions. Evidence is often the *same artifacts* organized differently per regulator.

**5. Monitor.** Models degrade. Data distributions shift. New regulations enact monthly. Continuous monitoring covers performance (accuracy, fairness, bias), safety (adversarial attacks, prompt injection), regulatory changes, and incidents.

**6. Improve.** Management review, quarterly or semi-annually, synthesizes monitoring data. What's not working? What changed in the regulatory landscape? Corrective actions feed back into controls.

### Roles

**Chief AI Officer (CAIO):**Emerging role, increasingly mandatory. Enterprise AI strategy, risk governance, regulatory alignment.**Compliance:**Regulatory mapping, evidence management, conformity assessments.**Legal:**Regulatory interpretation, legal risk assessment, regulator engagement.**Security:**OWASP controls, penetration testing, model access control.**ML Engineering:**Technical controls, testing, monitoring, documentation.

### Tooling

Traditional GRC platforms manage *documentation about* AI. Purpose-built AI governance platforms integrate into the AI lifecycle: they connect to ML environments, track testing and validation, pull performance metrics from production, automate evidence collection, and generate conformity assessment artifacts.

The difference is integration. A GRC tool asks "what AI systems do we have?" and you enter them manually. An AI governance platform queries your ML infrastructure directly. For multinational enterprises, purpose-built platforms scale because they automate regulatory mapping, evidence aggregation, and artifact generation across jurisdictions.

Explore: AI governance solutions · AI governance tools · Frameworks library · Platform overview

## Frequently Asked Questions About AI Compliance

**What is AI compliance?**
AI compliance is the practice of meeting legal, regulatory, and standards-based obligations that apply to AI systems across every jurisdiction where they are developed, deployed, or used. It encompasses regulatory requirements (like the EU AI Act), voluntary standards (like ISO 42001 and NIST AI RMF), and sector-specific rules.

**When does the EU AI Act come into force?**
The EU AI Act entered into force on August 1, 2024, with phased implementation. Prohibited practices took effect February 2, 2025; GPAI obligations August 2, 2025; general application (including high-risk) August 2, 2026. The Digital Omnibus may delay high-risk standalone system deadlines to December 2027.

**Does my US company need to comply with the EU AI Act?**
Yes, if your AI system is placed on the EU market or its output is used in the EU. The EU AI Act has extraterritorial reach. It applies to providers regardless of where they are established and to deployers of AI systems within the EU.

**What is ISO 42001 and do I need it?**
ISO/IEC 42001 is the international standard for AI management systems (published December 2023). It's certifiable and provides a structured governance framework. While not legally required by any regulation, certification is increasingly expected in enterprise procurement and demonstrates governance maturity to regulators.

**What's the difference between the EU AI Act and NIST AI RMF?**
The EU AI Act is a binding regulation with penalties up to 7% of global turnover; NIST AI RMF is a voluntary framework. They're complementary: NIST provides the risk management methodology, the EU AI Act defines the legal requirements. Most organizations use both.

**Is the Colorado AI Act still happening?**
As of April 2026, the Colorado AI Act (SB 24-205) is enacted but delayed to June 30, 2026 (originally February 1, 2026). Further amendments or repeal are under discussion, and the December 2025 federal preemption EO adds uncertainty.

**What is TRAIGA and who does it apply to?**
TRAIGA (Texas Responsible AI Governance Act, HB 149) has been in force since January 1, 2026. It applies to businesses deploying AI in Texas, requiring reasonable care, transparency, testing, and impact assessments. The Texas AG has exclusive enforcement with a 60-day cure period.

**What happened to SB 1047?**
California's SB 1047 was vetoed by Governor Newsom in September 2024, citing regulatory overreach. Its successor, SB 53 (Frontier AI Transparency Act), was signed September 2025 and is in force as of January 1, 2026.

**Is there a US federal AI law?**
No. As of April 2026, the US has no comprehensive federal AI legislation. Federal policy consists of executive orders, voluntary frameworks (NIST AI RMF), and sector guidance from the FDA, FTC, and FINRA.

**How long does AI compliance take to implement?**
Organizations with existing ISO 27001 or SOC 2 programs can typically extend to AI within 3 to 6 months. Building from scratch (inventory, controls, certification readiness) takes 6 to 12 months for a mid-size enterprise.

## What to Do Next

The regulatory landscape is complex, but the path forward is clear: inventory your systems, classify against applicable regulations, implement unified controls, and monitor continuously.

**See it in action.** Book a demo of the Modulos platform to see automated regulatory mapping, control frameworks, and continuous monitoring across jurisdictions. Book a demo →

**Go deeper on the EU AI Act.** Download our comprehensive guide covering risk classification, conformity assessment, and timelines. Download the EU AI Act guide →

**Explore the platform.** Browse the framework library, scoping tools, and compliance dashboards. Explore Modulos →

## Download the EU AI Act Guide

Learn how to ensure your AI systems comply with the EU AI Act. This guide provides a clear overview of the regulation, mandatory compliance requirements, and how to prepare your AI operations for these changes.

Download the Guide### EU AI Act Guide

Foundations and

Practical Insights

## Need Help Navigating Global AI Regulations?

Book a demo to see how the Modulos AI Governance Platform automates regulatory mapping, control frameworks, and continuous monitoring across jurisdictions.