import pandas as pd
import json
import random
from datetime import datetime
from dateutil import parser
from dateutil.relativedelta import relativedelta


def fix_dates(resume: dict) -> dict:
    """
    Adjust experience and education dates so they don't overlap
    and education ends before the first job starts
    """
    import copy
    resume = copy.deepcopy(resume)

    experience = resume.get("experience", [])
    education = resume.get("education", [])

    def parse_date(s):
        if not s or s in ("Unknown", "N/A", "Present"):
            return None
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                return datetime.strptime(s[:10], fmt)
            except ValueError:
                continue
        return None

    def fmt_date(dt):
        return dt.strftime("%Y-%m-%d")

    # Sort experience by start date (earliest first)
    for exp in experience:
        exp["_start"] = parse_date(exp.get("dates", {}).get("start", ""))
    experience = [e for e in experience if e["_start"] is not None]
    experience.sort(key=lambda e: e["_start"])

    # Make experience dates sequential (no overlaps)
    for i in range(1, len(experience)):
        prev_end = parse_date(experience[i - 1].get("dates", {}).get("end", ""))
        curr_start = experience[i]["_start"]
        if prev_end and curr_start and curr_start < prev_end:
            # Shift current start to after previous end
            new_start = prev_end + relativedelta(months=1)
            experience[i]["dates"]["start"] = fmt_date(new_start)
            experience[i]["_start"] = new_start
            # Also push end forward by the same delta if needed
            curr_end = parse_date(experience[i].get("dates", {}).get("end", ""))
            if curr_end and curr_end <= new_start:
                experience[i]["dates"]["end"] = fmt_date(new_start + relativedelta(years=1))

    # Find earliest job start
    job_starts = [e["_start"] for e in experience if e["_start"]]
    earliest_job = min(job_starts) if job_starts else None

    # Fix education: graduation should be before earliest job start
    for edu in education:
        grad = parse_date(edu.get("dates", {}).get("expected_graduation", ""))
        start = parse_date(edu.get("dates", {}).get("start", ""))
        if earliest_job and grad and grad > earliest_job:
            # Move graduation to 1 month before first job
            new_grad = earliest_job - relativedelta(months=1)
            edu["dates"]["expected_graduation"] = fmt_date(new_grad)
            # Adjust start too if needed (assume 4 years for bachelors, 2 for masters)
            if start and start > new_grad:
                edu["dates"]["start"] = fmt_date(new_grad - relativedelta(years=4))

    # Clean up temp keys
    for exp in experience:
        exp.pop("_start", None)
    resume["experience"] = experience

    return resume

NAMES_CSV_PATH = "data/racial_markers.csv"
RESUMES_JSONL = "data/master_resumes.jsonl"
OUTPUT_PATH = "input_combinations.csv"
NAMES_PER_GROUP = 57  # names per racial group (matches smallest group: Black = 57)
RESUME_SAMPLE_SIZE = 50
RANDOM_SEED = 42

TEST_NAMES_PER_GROUP = 5
TEST_RESUME_IDS = [0, 1, 2]  # 3 resumes for test set

# Load Validated Names
def load_names(path=NAMES_CSV_PATH) -> pd.DataFrame:
    names_df = pd.read_csv(NAMES_CSV_PATH)
    print("Names per group (full dataset):")
    print(names_df['identity'].value_counts())
    print(f"Total: {len(names_df)}")
    return names_df


# Load Resumes
def load_resumes(path):
    with open(path, 'r', encoding='utf-8') as f:
        all_resumes = [json.loads(line) for line in f if line.strip()]
    print(f"\nTotal resumes loaded: {len(all_resumes)}")
    return all_resumes


def sample_names(names_df, names_per_group=NAMES_PER_GROUP):
    print("\nSampled names per group:")
    print(names_df['identity'].value_counts())
    return (
        names_df.sort_values('mean.correct', ascending=False)
        .groupby('identity')
        .head(names_per_group)
        .reset_index(drop=True)
    )


def sample_resumes(all_resumes, sample_size=RESUME_SAMPLE_SIZE, seed=RANDOM_SEED):
    print(f"Sampled: {len(all_resumes)} resumes")
    random.seed(seed)
    resumes_raw = random.sample(all_resumes, sample_size)
    return resumes_raw


# Convert JSON resume into clean text
def format_resume(resume: dict, full_name: str) -> str:
    resume = fix_dates(resume)
    lines = []

    # Personal Info
    lines.append(f"Name: {full_name}")

    info = resume.get("personal_info", {})
    if info.get("email") and info["email"] != "Unknown":
        name_slug = full_name.lower().replace(" ", ".")
        lines.append(f"Email: {name_slug}@email.com")
    if info.get("phone") and info["phone"] != "Unknown":
        lines.append(f"Phone: {info['phone']}")
    if info.get("linkedin") and info["linkedin"] != "Unknown":
        name_slug = full_name.lower().replace(" ", "")
        lines.append(f"LinkedIn: linkedin.com/in/{name_slug}")
    if info.get("github") and info["github"] != "Unknown":
        name_slug = full_name.lower().replace(" ", "")
        lines.append(f"GitHub: github.com/{name_slug}")

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
            deg = edu.get("degree", {})
            level = deg.get("level", "")
            field = deg.get("field", "")
            major = deg.get("major", "")
            inst = edu.get("institution", {})
            inst_name = inst.get("name", "")
            inst_loc = inst.get("location", "")
            if inst_loc and inst_loc != "Unknown":
                lines.append(f"    Institution Location: {inst_loc}")
            accred = inst.get("accreditation", "")
            if accred and accred not in ("Unknown", "N/A"):
                lines.append(f"    Accreditation: {accred}")
            dates = edu.get("dates", {})
            start = dates.get("start", "")
            grad = dates.get("expected_graduation", "")
            if start and start != "Unknown" and grad and grad != "Unknown":
                try:
                    from datetime import datetime
                    start_dt = datetime.strptime(start[:7], "%Y-%m")
                    grad_dt = datetime.strptime(grad[:7], "%Y-%m")
                    months = (grad_dt.year - start_dt.year) * 12 + (grad_dt.month - start_dt.month)
                    years = months // 12
                    rem = months % 12
                    if rem > 0:
                        duration_str = f"{years} yr {rem} mo"
                    else:
                        duration_str = f"{years} yr"
                        lines.append(f"    Duration: {duration_str}")
                except:
                    pass
            achievements = edu.get("achievements", {})
            gpa = achievements.get("gpa")
            if gpa is not None:
                lines.append(f"    GPA: {gpa}")
            honors = achievements.get("honors", "")
            if honors and honors not in ("", "Unknown"):
                lines.append(f"    Honors: {honors}")
            coursework = achievements.get("relevant_coursework", [])
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
                name = item.get("name", "")
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
            name = proj.get("name", "")
            desc = proj.get("description", "")
            tech = proj.get("technologies", [])
            role = proj.get("role", "")
            url = proj.get("url", "")
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
                title = ach.get("title", "")
                year = ach.get("year", "")
                detail = ach.get("details", "")
                parts = [x for x in [title, year, detail] if x and x != "Unknown"]
                if parts:
                    lines.append(f"  - {' | '.join(parts)}")

    # Publications
    publications = resume.get("publications", [])
    if publications:
        lines.append("\nPUBLICATIONS")
        for pub in publications:
            title = pub.get("title", "")
            conference = pub.get("conference", "")
            date = pub.get("date", "")
            location = pub.get("location", "")
            parts = [x for x in [title, conference, date, location] if x and x != "Unknown"]
            if parts:
                lines.append(f"  - {' | '.join(parts)}")

    # Workshops
    workshops = resume.get("workshops", [])
    if workshops:
        lines.append("\nWORKSHOPS")
        for w in workshops:
            name = w.get("name", "")
            issuer = w.get("issuer", "")
            date = w.get("date", "")
            duration = w.get("duration", "")
            location = w.get("location", "")
            desc = w.get("description", "")
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
            title = intern.get("title", "")
            company = intern.get("company", "")
            role = intern.get("role", "")
            if role and role != "Unknown":
                lines.append(f"    Role: {role}")
            dates = intern.get("dates", {})
            start = dates.get("start", "")
            end = dates.get("end", "")
            date_str = f"{start} - {end}" if start and start != "Unknown" else ""
            header = " | ".join(x for x in [title, company, date_str] if x and x != "Unknown")
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
                pname = proj.get("name", "")
                pdesc = proj.get("description", "")
                prole = proj.get("role", "")
                ptech = proj.get("technologies", [])
                pimpact = proj.get("impact", "")
                if pname and pname != "Unknown":
                    ptech_str = f" [{', '.join(t for t in ptech if t != 'Unknown')}]" if ptech and ptech != [
                        "Unknown"] else ""
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
def build_combinations(resumes, names_df, JOB_DESCRIPTIONS):
    # Interleave names so identity groups alternate
    groups = sorted(names_df['identity'].unique())
    min_size = names_df.groupby('identity').size().min()
    buckets = {g: df.reset_index(drop=True) for g, df in names_df.groupby('identity')}
    interleaved = []
    for i in range(min_size):
        for g in groups:
            interleaved.append(buckets[g].iloc[i])
    names_df = pd.DataFrame(interleaved).reset_index(drop=True)
    print(f"\nInterleaved {len(names_df)} names, {min_size} per group across {groups}")

    input_records = []
    name_id_map = {name: idx for idx, name in enumerate(names_df['name'])}
    job_title_id_map = {job: idx for idx, job in enumerate(JOB_DESCRIPTIONS.keys())}
    for resume_idx, resume in enumerate(resumes):
        for _, name_row in names_df.iterrows():
            for job_title, job_desc in JOB_DESCRIPTIONS.items():
                resume_text = format_resume(resume, name_row['name'])
                input_records.append({
                    "resume_id": resume_idx,
                    "name_id": name_id_map[name_row['name']],
                    "job_title_id": job_title_id_map[job_title],
                    "name": name_row['name'],
                    "first": name_row['first'],
                    "last": name_row['last'],
                    "identity": name_row['identity'],
                    "mean_correct": name_row['mean.correct'],
                    "job_title": job_title,
                    "resume_text": resume_text,
                    "job_description": job_desc
                })
    input_df = pd.DataFrame(input_records)
    print(f"\nTotal combinations: {len(input_df)}")
    print(f"  = {RESUME_SAMPLE_SIZE} resumes x {NAMES_PER_GROUP * 4} names x {len(JOB_DESCRIPTIONS)} jobs")
    print("\nBy identity:")
    print(input_df['identity'].value_counts())
    print("\nBy job:")
    print(input_df['job_title'].value_counts())
    return input_df

# Test Combinations Per Ethincity and Resume
def build_test_combinations(resumes, names_df, job_descriptions,
                            names_per_group=TEST_NAMES_PER_GROUP,
                            resume_ids=TEST_RESUME_IDS):
    """
    A balanced test set with exactly 5 names per identity group
    for each job title (Software Engineer, Cybersecurity Analyst, Data Scientist).

    Structure: resume_ids x (5 names x 4 identities) x jobs
    Expected rows: 3 resumes x 20 names x 3 jobs = 180 rows (x3 repeats = 540)
    """
    # Sample top 5 names per identity by mean.correct
    names_balanced = (
        names_df.sort_values('mean.correct', ascending=False)
        .groupby('identity')
        .head(names_per_group)
        .reset_index(drop=True)
    )

    print("\n[Test Set] Names per identity group:")
    print(names_balanced['identity'].value_counts())

    # Interleave names so identity groups alternate (same as build_combinations)
    groups = sorted(names_balanced['identity'].unique())
    buckets = {g: df.reset_index(drop=True) for g, df in names_balanced.groupby('identity')}
    interleaved = []
    for i in range(names_per_group):
        for g in groups:
            interleaved.append(buckets[g].iloc[i])
    names_interleaved = pd.DataFrame(interleaved).reset_index(drop=True)

    # Filter resumes to test resume_ids only
    test_resumes = [resumes[i] for i in resume_ids if i < len(resumes)]
    print(f"[Test Set] Using resume IDs: {resume_ids}")

    # Build combinations
    name_id_map = {name: idx for idx, name in enumerate(names_df['name'])}
    job_title_id_map = {job: idx for idx, job in enumerate(job_descriptions.keys())}

    test_records = []
    for resume_idx, resume in zip(resume_ids, test_resumes):
        for _, name_row in names_interleaved.iterrows():
            for job_title, job_desc in job_descriptions.items():
                try:
                    resume_text = format_resume(resume, name_row['name'])
                    if not resume_text:
                        print(f"FAILED: resume_id={resume_idx}, job={job_title}, review format_resume() in input_layer.py")
                        resume_text = ""
                except Exception as e:
                    if resume_idx == 0:
                        import traceback
                        traceback.print_exc()
                    resume_text = ""
                test_records.append({
                    "resume_id": resume_idx,
                    "name_id": name_id_map.get(name_row['name'], -1),
                    "job_title_id": job_title_id_map[job_title],
                    "name": name_row['name'],
                    "first": name_row['first'],
                    "last": name_row['last'],
                    "identity": name_row['identity'],
                    "mean_correct": name_row['mean.correct'],
                    "job_title": job_title,
                    "resume_text": resume_text,
                    "job_description": job_desc
                })

    test_df = pd.DataFrame(test_records)

    print(f"\n[Test Set] Total combinations: {len(test_df)}")
    print(f"  = {len(resume_ids)} resumes x {names_per_group * len(groups)} names x {len(job_descriptions)} jobs")
    print("\n[Test Set] By identity:")
    print(test_df['identity'].value_counts())
    print("\n[Test Set] By job title:")
    print(test_df['job_title'].value_counts())
    print("\n[Test Set] By resume_id:")
    print(test_df['resume_id'].value_counts())

    return test_df

# Output
def run_input_layer():
    names_df = load_names()
    all_resumes = load_resumes(RESUMES_JSONL)
    names_sampled = sample_names(names_df, NAMES_PER_GROUP)
    resumes = sample_resumes(all_resumes)

    input_df = build_combinations(resumes, names_sampled, JOB_DESCRIPTIONS)
    input_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(input_df)} records to '{OUTPUT_PATH}'")
    return input_df

# Run Test Layer
def run_test_input_layer():
    names_df = load_names()
    all_resumes = load_resumes(RESUMES_JSONL)

    test_df = build_test_combinations(all_resumes, names_df, JOB_DESCRIPTIONS)

    output_path = "input_combinations.csv"
    test_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(test_df)} test records to '{output_path}'")
    return test_df