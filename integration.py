from concatfiles import main_process  # your data processing function
from chunks import answer_query  # your NetCDF QA function
from analyze import data_transformation  # your data transformation function
from sematics import run_semantics  # your semantics function
import os

from dbagent import query_db_q,initialize_db

from chunks import answer_query,save_embeddings
def integration(uuid,user_folder,argo_folder):
    output_path = os.path.join(user_folder, "argo_profiles_binned_1.csv")
    main_process(save_path=output_path,nc_data_dir=os.path.join(argo_folder,"*.nc"))
    processed_output_path=os.path.join(user_folder, "argo_profiles_final.csv")
    data_transformation(input_path=output_path, output_path=processed_output_path)
    print("Data processing complete. Now semantic menaning will be added")
    semantic_output_path=os.path.join(user_folder, "argo_semantic_summary_1.csv")
    run_semantics(input_path=processed_output_path, output_path=semantic_output_path)
    # save it in embeddings
    save_embeddings(input_path=semantic_output_path, user_id=uuid)

    #save it in data base
    conn=initialize_db(input_path=semantic_output_path, db_path=user_folder,user_id=uuid)
    return semantic_output_path,conn

def netcdf_qa(question, user_id=None,semantic_output_path=None):
    response = answer_query(question,user_id=user_id)
    return response

def sql_qa(question,conn):
    df=query_db_q(question,conn)
    return df