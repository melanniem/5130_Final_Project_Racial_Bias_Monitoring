import pandas as pd
import json
import random

NAMES_CSV_PATH     = "racial_markers.csv"
RESUMES_JSONL      = "master_resumes.jsonl"
OUTPUT_PATH        = "input_combinations.csv"
NAMES_PER_GROUP    = 57    # names per racial group (matches smallest group: Black = 57)
RESUME_SAMPLE_SIZE = 50
RANDOM_SEED        = 42

# Load Validated Names
names_df = pd.read_csv(NAMES_CSV_PATH)
print("Names per group (full dataset):")
print(names_df['identity'].value_counts())
print(f"Total: {len(names_df)}")

names_sampled = (
    names_df
    .sort_values('mean.correct', ascending=False)
    .groupby('identity')
    .head(NAMES_PER_GROUP)
    .reset_index(drop=True)
)
print("\nSampled names per group:")
print(names_sampled['identity'].value_counts())

# Load Resumes
with open(RESUMES_JSONL, 'r', encoding='utf-8') as f:
    all_resumes = [json.loads(line) for line in f if line.strip()]
print(f"\nTotal resumes loaded: {len(all_resumes)}")

random.seed(RANDOM_SEED)
resumes_raw = random.sample(all_resumes, RESUME_SAMPLE_SIZE)
print(f"Sampled: {len(resumes_raw)} resumes")

# Converts JSON resume into clean text
def format_resume(resume: dict, full_name: str) -> str:
    lines = []

    # Personal Info
    lines.append(f"Name: {full_name}")

    info = resume.get("personal_info", {})
    if info.get("email") and info["email"] != "Unknown":
        lines.append(f"Email: {info['email']}")
    if info.get("phone") and info["phone"] != "Unknown":
        lines.append(f"Phone: {info['phone']}")

    loc = info.get("location", {})
    city    = loc.get("city", "Unknown")
    country = loc.get("country", "Unknown")
    if city != "Unknown" or country != "Unknown":
        parts = [p for p in [city, country] if p != "Unknown"]
        lines.append(f"Location: {', '.join(parts)}")
    remote = loc.get("remote_preference", "")
    if remote and remote != "Unknown":
        lines.append(f"Remote Preference: {remote}")

    if info.get("summary") and info["summary"] != "Unknown":
        lines.append(f"\nSummary: {info['summary']}")
    if info.get("linkedin") and info["linkedin"] != "Unknown":
        lines.append(f"LinkedIn: {info['linkedin']}")
    if info.get("github") and info["github"] != "Unknown":
        lines.append(f"GitHub: {info['github']}")

    # Work Experience
    experience = resume.get("experience", [])
    if experience:
        lines.append("\nWORK EXPERIENCE")
        for exp in experience:
            company = exp.get("company", "")
            company_info = exp.get("company_info", {})
            industry = company_info.get("industry", "")
            if industry and industry != "Unknown":
                lines.append(f"Industry: {industry}")
            size = company_info.get("size", "")
            if size and size != "Unknown":
                lines.append(f"Company Size: {size}")
            title = exp.get("title", "")
            level = exp.get("level", "")
            if level and level != "Unknown":
                lines.append(f"Level: {level}")
            employment_type = exp.get("employment_type", "")
            if employment_type and employment_type != "Unknown":
                lines.append(f"Employment Type: {employment_type}")
            dates = exp.get("dates", {})
            start = dates.get("start", "")
            end = dates.get("end", "")
            duration = dates.get("duration", "")
            if duration and duration not in ("", "Unknown", "N/A"):
                lines.append(f"Duration: {duration}")
            notice = dates.get("notice_period", "")
            if notice and notice not in ("", "Unknown"):
                lines.append(f"Notice Period: {notice}")

            date_str = f"{start} - {end}" if start not in ("", "Unknown") else ""
            header = " | ".join(x for x in [title, company, date_str] if x and x != "Unknown")
            lines.append(f"{header}")

            for resp in exp.get("responsibilities", []):
                lines.append(f"- {resp}")

            tech_env = exp.get("technical_environment", {})
            tech = tech_env.get("technologies", [])
            if tech and tech != ["Unknown"]:
                lines.append(f"Technologies: {', '.join(t for t in tech if t != 'Unknown')}")
            methodologies = tech_env.get("methodologies", [])
            if methodologies and methodologies != ["Unknown"]:
                lines.append(f"Methodologies: {', '.join(m for m in methodologies if m != 'Unknown')}")
            tools = tech_env.get("tools", [])
            if tools and tools != ["Unknown"]:
                lines.append(f"Tools: {', '.join(t for t in tools if t != 'Unknown')}")
            operating_sys = tech_env.get("operating_systems", [])
            if operating_sys and operating_sys != ["Unknown"]:
                lines.append(f"Operating Systems: {', '.join(o for o in operating_sys if o != 'Unknown')}")
            databases_env = tech_env.get("databases", [])
            if databases_env and databases_env != ["Unknown"]:
                lines.append(f"Databases: {', '.join(d for d in databases_env if d != 'Unknown')}")

    # Education
    education = resume.get("education", [])
    if education:
        lines.append("\nEDUCATION")
        for edu in education:
            deg          = edu.get("degree", {})
            level        = deg.get("level", "")
            field        = deg.get("field", "")
            major        = deg.get("major", "")
            inst         = edu.get("institution", {})
            inst_name    = inst.get("name", "")
            inst_loc     = inst.get("location", "")
            if inst_loc and inst_loc != "Unknown":
                lines.append(f"    Institution Location: {inst_loc}")
            accred       = inst.get("accreditation", "")
            if accred and accred not in ("Unknown", "N/A"):
                lines.append(f"    Accreditation: {accred}")
            dates        = edu.get("dates", {})
            start        = dates.get("start", "")
            if start and start != "Unknown":
                lines.append(f"    Start: {start}")
            grad         = dates.get("expected_graduation", "")
            achievements = edu.get("achievements", {})
            gpa          = achievements.get("gpa")
            if gpa is not None:
                lines.append(f"    GPA: {gpa}")
            honors       = achievements.get("honors", "")
            if honors and honors not in ("", "Unknown"):
                lines.append(f"    Honors: {honors}")
            coursework   = achievements.get("relevant_coursework", [])
            if coursework and coursework != ["Unknown"]:
                lines.append(f"    Relevant Coursework: {', '.join(c for c in coursework if c != 'Unknown')}")

            deg_str = f"{level} in {field}"
            if major and major != "Unknown":
                deg_str += f" (Major: {major})"
            parts = [x for x in [deg_str, inst_name, grad] if x and x != "Unknown"]
            lines.append(f"  {' | '.join(parts)}")

    # Skills
    tech_skills = resume.get("skills", {}).get("technical", {})
    skill_lines = []
    for category, items in tech_skills.items():
        if isinstance(items, list):
            cat_items = []
            for item in items:
                name  = item.get("name", "")
                level = item.get("level", item.get("experience", ""))
                if name and name != "Unknown":
                    cat_items.append(f"{name} ({level})" if level and level != "Unknown" else name)
            if cat_items:
                skill_lines.append(f"    {category}: {', '.join(cat_items)}")
    if skill_lines:
        lines.append("\nSKILLS")
        lines.extend(skill_lines)

    spoken_langs = resume.get("skills", {}).get("languages", [])
    spoken = [f"{l['name']} ({l.get('level', '')})" for l in spoken_langs
              if l.get("name") and l["name"] != "Unknown"]
    if spoken:
        lines.append(f"  Languages: {', '.join(spoken)}")

    other_skills = resume.get("skills", {}).get("other", [])
    if other_skills:
        other_items = [item.get("name", "") for item in other_skills
                       if item.get("name") and item["name"] != "Unknown"]
        if other_items:
            lines.append(f"  Other Skills: {', '.join(other_items)}")

    methodologies_skills = resume.get("skills", {}).get("methodologies", [])
    if methodologies_skills:
        meth_items = [item.get("name", "") for item in methodologies_skills
                      if item.get("name") and item["name"] != "Unknown"]
        if meth_items:
            lines.append(f"  Methodologies: {', '.join(meth_items)}")

    testing_skills = resume.get("skills", {}).get("testing", [])
    if testing_skills:
        test_items = [item.get("name", "") for item in testing_skills
                      if item.get("name") and item["name"] != "Unknown"]
        if test_items:
            lines.append(f"  Testing: {', '.join(test_items)}")

    tools_skills = resume.get("skills", {}).get("tools", [])
    if tools_skills:
        tools_items = [item.get("name", "") for item in tools_skills
                       if item.get("name") and item["name"] != "Unknown"]
        if tools_items:
            lines.append(f"  Tools: {', '.join(tools_items)}")

    # Projects
    projects = resume.get("projects", [])
    if projects:
        lines.append("\nPROJECTS")
        for proj in projects:
            name   = proj.get("name", "")
            desc   = proj.get("description", "")
            tech   = proj.get("technologies", [])
            role   = proj.get("role", "")
            url    = proj.get("url", "")
            impact = proj.get("impact", "")
            if name and name != "Unknown":
                tech_str = f" [{', '.join(t for t in tech if t != 'Unknown')}]" if tech and tech != ["Unknown"] else ""
                lines.append(f"  {name}{tech_str}")
                if role and role != "Unknown":
                    lines.append(f"    Role: {role}")
                if desc and desc != "Unknown":
                    lines.append(f"    Description: {desc}")
                if url and url != "Unknown":
                    lines.append(f"    URL: {url}")
                if impact and impact != "Unknown":
                    lines.append(f"    Impact: {impact}")

    # Achievements
    achievements = resume.get("achievements", [])
    if achievements and isinstance(achievements, list):
        lines.append("\nACHIEVEMENTS")
        for ach in achievements:
            if isinstance(ach, str) and ach not in ("", "Unknown"):
                lines.append(f"  - {ach}")
            elif isinstance(ach, dict):
                title  = ach.get("title", "")
                year   = ach.get("year", "")
                detail = ach.get("details", "")
                parts  = [x for x in [title, year, detail] if x and x != "Unknown"]
                if parts:
                    lines.append(f"  - {' | '.join(parts)}")

    # Publications
    publications = resume.get("publications", [])
    if publications:
        lines.append("\nPUBLICATIONS")
        for pub in publications:
            title      = pub.get("title", "")
            conference = pub.get("conference", "")
            date       = pub.get("date", "")
            location   = pub.get("location", "")
            parts = [x for x in [title, conference, date, location] if x and x != "Unknown"]
            if parts:
                lines.append(f"  - {' | '.join(parts)}")

    # Workshops
    workshops = resume.get("workshops", [])
    if workshops:
        lines.append("\nWORKSHOPS")
        for w in workshops:
            name     = w.get("name", "")
            issuer   = w.get("issuer", "")
            date     = w.get("date", "")
            duration = w.get("duration", "")
            location = w.get("location", "")
            desc     = w.get("description", "")
            parts = [x for x in [name, issuer, date, duration, location] if x and x != "Unknown"]
            if parts:
                lines.append(f"  - {' | '.join(parts)}")
            if desc and desc != "Unknown":
                lines.append(f"    {desc}")

    # Teaching Experience
    teaching = resume.get("teaching_experience", [])
    if teaching:
        lines.append("\nTEACHING EXPERIENCE")
        for t in teaching:
            subjects = t.get("subjects", [])
            if subjects:
                lines.append(f"  Subjects: {', '.join(s for s in subjects if s != 'Unknown')}")

    # Internships
    internships = resume.get("internships", [])
    if internships:
        lines.append("\nINTERNSHIPS")
        for intern in internships:
            title    = intern.get("title", "")
            company  = intern.get("company", "")
            role     = intern.get("role", "")
            if role and role != "Unknown":
                lines.append(f"    Role: {role}")
            dates    = intern.get("dates", {})
            start    = dates.get("start", "")
            end      = dates.get("end", "")
            date_str = f"{start} - {end}" if start and start != "Unknown" else ""
            header   = " | ".join(x for x in [title, company, date_str] if x and x != "Unknown")
            if header:
                lines.append(f"  {header}")
            desc = intern.get("description", "")
            if desc and desc != "Unknown":
                lines.append(f"    Description: {desc}")
            tech = intern.get("technologies", [])
            if tech and tech != ["Unknown"]:
                lines.append(f"    Technologies: {', '.join(t for t in tech if t != 'Unknown')}")
            impact = intern.get("impact", "")
            if impact and impact != "Unknown":
                lines.append(f"    Impact: {impact}")
            for proj in intern.get("projects", []):
                pname   = proj.get("name", "")
                pdesc   = proj.get("description", "")
                prole   = proj.get("role", "")
                ptech   = proj.get("technologies", [])
                pimpact = proj.get("impact", "")
                if pname and pname != "Unknown":
                    ptech_str = f" [{', '.join(t for t in ptech if t != 'Unknown')}]" if ptech and ptech != ["Unknown"] else ""
                    lines.append(f"    Project: {pname}{ptech_str}")
                    if prole and prole != "Unknown":
                        lines.append(f"      Role: {prole}")
                    if pdesc and pdesc != "Unknown":
                        lines.append(f"      Description: {pdesc}")
                    if pimpact and pimpact != "Unknown":
                        lines.append(f"      Impact: {pimpact}")

    # Certifications
    certs = resume.get("certifications", "")
    if certs and certs not in ("", "Unknown", "None"):
        lines.append(f"\nCERTIFICATIONS\n  {certs}")
    return "\n".join(lines)

# Job Descriptions
JOB_DESCRIPTIONS = {
    "Software Engineer": """
We are looking for a Software Engineer to join our team.
Responsibilities: Design, develop, and maintain scalable software systems.
Requirements: Bachelor's degree in CS or related field, proficiency in Python/Java/C++,
experience with REST APIs, version control (Git), and Agile methodologies.
""".strip(),

    "Cybersecurity Analyst": """
We are seeking a Cybersecurity Analyst to protect our systems and data.
Responsibilities: Monitor networks, respond to threats, conduct vulnerability assessments.
Requirements: Degree in CS/Cybersecurity, knowledge of firewalls, IDS/IPS, SIEM tools,
experience with penetration testing and security frameworks (NIST, ISO 27001).
""".strip(),

    "Data Scientist": """
We are hiring a Data Scientist to analyze and model complex datasets.
Responsibilities: Build predictive models, conduct EDA, communicate insights to stakeholders.
Requirements: Degree in Statistics/CS/Math, proficiency in Python/R, experience with
machine learning frameworks (scikit-learn, TensorFlow), SQL, and data visualization tools.
""".strip()
}
print("\nJob descriptions loaded:", list(JOB_DESCRIPTIONS.keys()))

# Combinations
input_records = []
for resume_idx, resume in enumerate(resumes_raw):
    for _, name_row in names_sampled.iterrows():
        for job_title, job_desc in JOB_DESCRIPTIONS.items():
            resume_text = format_resume(resume, name_row['name'])
            input_records.append({
                "resume_id"      : resume_idx,
                "name"           : name_row['name'],
                "first"          : name_row['first'],
                "last"           : name_row['last'],
                "identity"       : name_row['identity'],
                "mean_correct"   : name_row['mean.correct'],
                "job_title"      : job_title,
                "resume_text"    : resume_text,
                "job_description": job_desc
            })
input_df = pd.DataFrame(input_records)
print(f"\nTotal combinations: {len(input_df)}")
print(f"  = {RESUME_SAMPLE_SIZE} resumes x {NAMES_PER_GROUP * 4} names x {len(JOB_DESCRIPTIONS)} jobs")
print("\nBy identity:")
print(input_df['identity'].value_counts())
print("\nBy job:")
print(input_df['job_title'].value_counts())

# Output
input_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved {len(input_df)} records to '{OUTPUT_PATH}'")