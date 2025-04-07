import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_option_menu import option_menu

#import required Datasets
@st.cache_data
def datasets():
    role_dataset = pd.read_csv(r"Role_description.csv")
    skills_dataset = pd.read_excel(r"skills_dataset.xlsx")
    job_dataset = pd.read_csv(r"Job.csv")
    resume_dataset =pd.read_csv(r"Resume_dataset.csv")
    qualification_dataset = pd.read_csv(r"qualification.csv")
    country_dataset = pd.read_csv(r"country.csv")

    return role_dataset,skills_dataset,job_dataset,resume_dataset,qualification_dataset,country_dataset



role_dataset,skills_dataset,job_dataset,resume_dataset,qualification_dataset,country_dataset = datasets()

@st.cache_data
def load_and_vectorize():
    role_tf_idf = TfidfVectorizer()
    Employee_tf_idf = TfidfVectorizer()
    role_dataset['Text'] = role_dataset['Text'].fillna('').astype(str)
    resume_dataset['recommend_desc'] = resume_dataset['recommend_desc'].fillna('').astype(str)
    role_vector = role_tf_idf.fit_transform(role_dataset["Text"])
    Employee_vector = Employee_tf_idf.fit_transform(resume_dataset['recommend_desc'])
    return role_tf_idf,Employee_tf_idf, role_vector , Employee_vector


def recommend(desc , mode):
    recommendation = []
    role_tf_idf,Employee_tf_idf, role_vector,Employee_vector = load_and_vectorize()
    
    if mode == 'Job Recommendation':
        description_vector = role_tf_idf.transform(desc)
        similarity = cosine_similarity(role_vector , description_vector).flatten()
        sorted_list = sorted(list(enumerate(similarity)),reverse= True , key= lambda x : x[1])
        for i in sorted_list:
            if i[1] >0.2:
                if role_dataset.iloc[i[0]].Category not in recommendation:
                    # st.write(role_dataset.iloc[i[0]].Category)
                    recommendation.append(role_dataset.iloc[i[0]]["Category"])
    if mode == 'Employee Recommendation':
        description_vector = Employee_tf_idf.transform(desc)
        similarity = cosine_similarity(Employee_vector , description_vector).flatten()
        sorted_list = sorted(list(enumerate(similarity)) , reverse=True , key = lambda x : x[1])
        for  i in sorted_list:
            Employee_index = i[0]
            if i[1]>0.2:  #Return the similarity grater than 0.2 index
                recommendation.append(Employee_index) 
    return recommendation

with st.sidebar:
    selected = option_menu("Main Menu " , ['Admin Panel' , 'Job Recommendation' , 'Employee Recommendation' , ] , icons=[ 'database-add' , 'briefcase' ,'person'] , menu_icon = 'house-door' )

# UI for About Project section
if selected == 'About Project':
    st.title("About Project")

# UI for Job Recommendation Section
if selected == 'Job Recommendation':
    st.title("Job Recommendation") 
    with st.form('Job-Find'):
        description=st.text_area("Objactive / Profile Summery")
        skills = st.multiselect('skills' , skills_dataset['Skills'].unique())
        skills = ' '.join(skills)
        # experience = st.selectbox("Experience " , job_dataset['Experience'].unique())
        # qualification = st.selectbox("Qualification" , job_dataset['Qualifications'].unique())
        recommended = st.form_submit_button("Recommend")
        description = description.lower() +" "+ skills.lower()
    # Recommend Button Activity

    if recommended:
        mode = "Job Recommendation"
        roles = recommend([description] , mode)

        for i in roles:
            job_data = job_dataset[(job_dataset['Role'] == i) ] #& (job_dataset['Experience'] == experience)&(job_dataset['Qualifications'] == qualification)]
            st.write(f"## {i} Jobs")
            if job_data.empty:
                st.error(f'No Jobs Available in {i} role')
                continue
            container1 = st.container(border=True)
            for j in range(0 , len(job_data)):
                with container1:
                    st.write(f"### Company {j+1}")
                    st.write(f"##### Company Name ")
                    st.write(job_data.iloc[j,10])
                    col1 , col2 , col3= st.columns(3) 
                    st.write(f"##### Responsiblity ")
                    st.write(job_data.iloc[j,9])
                    st.write(f"##### Skill Requirement")
                    st.write(job_data.iloc[j,8])
                    st.write("_"*40)
                    col1.write(f"##### Role")
                    col1.write(job_data.iloc[j,6])
                    col2.write(f"##### Expirence Requirment")
                    col2.write(job_data.iloc[j,1].astype(str))
                    col3.write(f"##### Country")
                    col3.write(job_data.iloc[j,3])
                    col1.write(f"##### Contact Person")
                    col1.write(job_data.iloc[j,4])
                    col2.write(f"##### Contact Number")
                    col2.write(job_data.iloc[j,5].astype(str))
                    col3.write(f"##### Company Mail")
                    col3.write(job_data.iloc[j,-1])

# UI for Employee Recommendation

if selected == 'Employee Recommendation':
    st.title("Employee Recommendation")
    with st.form("Employee Recommendation"):
        description=st.text_area("Job Description")
        skills = st.multiselect('Skill Requirement' , skills_dataset['Skills'].unique())
        skills = ' '.join (skills)
        recommended = st.form_submit_button('Recommend')
        description = description.lower() + skills.lower()

    if recommended:
        mode = 'Employee Recommendation'
        Employee_index = recommend([description] , mode)
        st.write(f"## Employees")
        for i,j in enumerate(Employee_index):
            container1 = st.container(border=True)
            with container1:
                st.write(f"### Employee {i+1}")
                st.write(f"##### Employee Name ")
                st.write(resume_dataset.iloc[j,0])
                col1 , col2 , col3= st.columns(3) 
                st.write(f"##### Description ")
                st.write(resume_dataset.iloc[j,4])
                st.write("_"*40)
                col1.write(f"##### Role")
                col1.write(resume_dataset.iloc[j,5])
                col2.write(f"##### Phone Number")
                col2.write(resume_dataset.iloc[j,6])
                col3.write(f"##### Email ")
                col3.write(resume_dataset.iloc[j,7])
                col1.write(f"##### Qualification")
                col1.write(resume_dataset.iloc[j,1])


# UI for Admin Panel
if selected == 'Admin Panel':
    Option = option_menu(None , ["Add Employee" , "Add Job"],icons=['person' , 'briefcase'],default_index=0, orientation="horizontal")
    if Option == 'Add Employee':
        with st.form("Add Emplyee"):
            name = st.text_input('Name')
            qualification = st.selectbox("Qualification" , qualification_dataset['Qualification'])
            skills = st.multiselect('skills' , skills_dataset['Skills'].unique())
            skills = ' '.join(skills)
            role = st.text_input('Role')
            experience = st.slider("Years of Experience", 0, 30, 1)
            description=st.text_area("Objactive / Profile Summery")
            contact = st.text_input("Contact No")
            email = st.text_input("Email Address")
            recommend_desc = description.lower() + skills.lower() + role.lower()
            add_employee = st.form_submit_button("Add")

            if add_employee :
                Employee = {
                    'name': name , 
                    'qualification' : qualification , 
                    'skills' : [skills] , 
                    'experience' : experience , 
                    'description' : description , 
                    'role' : role ,
                    'contact' : contact,
                    'email' : email,
                    'recommend_desc': recommend_desc
                }
                Employee = pd.DataFrame(Employee)

                resume_dataset = pd.concat([resume_dataset , Employee])
                resume_dataset.to_csv(r"C:\Project\Role-Recommendation\datasets\Resume_dataset.csv", index=False)
                # st.dataframe(resume_dataset)
                st.success('Employee Added Successfully')
                datasets.clear()
                load_and_vectorize.clear()
                st.rerun()

    if Option == 'Add Job':
        with st.form("Add Emplyee"):
            company = st.text_input('Company Name')
            description=st.text_area("Job Summary")
            country = st.selectbox("Country" ,sorted(country_dataset['Country']) )
            qualification = st.selectbox("Qualification Reuqired" , qualification_dataset['Qualification'])
            skills = st.multiselect('Skills Requirement' , skills_dataset['Skills'].unique())
            Role = st.text_input("Role")
            responsiblity = st.text_area("Role Responsiblity")
            experience = st.slider("Years of Experience Requuired", 0, 30, 1)
            contact = st.text_input("Contact No")
            email = st.text_input("Company Mail Address")
            contact_person = st.text_input("Hiring Person name / contact person")

            add_job = st.form_submit_button("Add")
            skills = ','.join(skills)


            if add_job :
                job = {
                    'Experience':experience,
                    'Qualifications':qualification,
                    'Country':country,
                    'Contact Person':contact_person,
                    'Contact':contact,
                    'Role':Role,
                    'Job Description':description,
                    'skills': skills,
                    'Responsibilities':responsiblity,
                    'Company':company,
                    'Mail':email
                }
                st.write([job])
                Job = pd.DataFrame([job])
                Role_description = pd.DataFrame([job])
                Role_description['Description'] = Role_description['Job Description'] + Role_description['skills'] + Role_description['Responsibilities'] +Role_description['Role']
                Role_description = Role_description[['Role' , 'Description']] 
                Role_description['Description'] = Role_description['Description'].apply(lambda x : x.lower())
                Role_description.columns = ['Category' , 'Text']
                role_dataset = pd.concat([role_dataset , Role_description] , ignore_index = True)
                job_dataset = pd.concat([job_dataset , Job] , ignore_index= True)
                job_dataset.to_csv(r"C:\Project\Role-Recommendation\datasets\Job.csv", index=False)
                role_dataset.to_csv(r"C:\Project\Role-Recommendation\datasets\Role_description.csv" , index=False)
                # st.dataframe(resume_dataset)
                st.success('Job Added Successfully')
                datasets.clear()
                load_and_vectorize.clear()
                st.rerun()