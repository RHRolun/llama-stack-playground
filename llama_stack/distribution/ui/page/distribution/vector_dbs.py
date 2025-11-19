# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import streamlit as st

from ...modules.api import llama_stack_api


def vector_dbs():
    """
    Inspect available vector stores and display details for the selected one.
    """
    st.header("Vector Stores")
    # Fetch all vector stores using modern API
    vector_stores = llama_stack_api.client.vector_stores.list()
    if not vector_stores.data:
        st.info("No vector stores found.")
        return
    # Build info dict and allow selection
    vdb_info = {v.id: v.dict() for v in vector_stores.data}
    selected_vector_store = st.selectbox("Select a vector store", list(vdb_info.keys()))
    st.json(vdb_info[selected_vector_store], expanded=True)
