// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <gtest/gtest.h>
#include <src/index/knowhere/knowhere/index/vector_index/helpers/IndexParameter.h>
#include <iostream>
#include <sstream>

#include <NGT/Common.h>
#include "knowhere/common/Exception.h"
#include "knowhere/index/vector_index/IndexNGTONNG.h"

#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class NGTONNGTest : public DataGen, public TestWithParam<std::string> {
 protected:
    void
    SetUp() override {
        IndexType = GetParam();
        // Generate(128, 10000, 10);
        GenerateSift1M(128, 1000000, 10000);
        index_ = std::make_shared<milvus::knowhere::IndexNGTONNG>();
        conf = milvus::knowhere::Config{
            {milvus::knowhere::meta::DIM, dim},
            {milvus::knowhere::meta::TOPK, 100},
            {milvus::knowhere::Metric::TYPE, milvus::knowhere::Metric::L2},
            {milvus::knowhere::IndexParams::edge_size, 20},
            {milvus::knowhere::IndexParams::outgoing_edge_size, 5},
            {milvus::knowhere::IndexParams::incoming_edge_size, 40},
        };
    }

 protected:
    milvus::knowhere::Config conf;
    std::shared_ptr<milvus::knowhere::IndexNGTONNG> index_ = nullptr;
    std::string IndexType;
};

INSTANTIATE_TEST_CASE_P(NGTONNGParameters, NGTONNGTest, Values("NGTONNG"));

TEST_P(NGTONNGTest, ngtonng_basic) {
    assert(!xb.empty());

    // null index
#if 0
    {
        ASSERT_ANY_THROW(index_->Train(base_dataset, conf));
        ASSERT_ANY_THROW(index_->Query(query_dataset, conf));
        ASSERT_ANY_THROW(index_->Serialize(conf));
        ASSERT_ANY_THROW(index_->Add(base_dataset, conf));
        ASSERT_ANY_THROW(index_->AddWithoutIds(base_dataset, conf));
        ASSERT_ANY_THROW(index_->Count());
        ASSERT_ANY_THROW(index_->Dim());
    }

#endif
    index_->BuildAll(base_dataset, conf);  // Train + Add
    // ASSERT_EQ(index_->Count(), nb);
    // ASSERT_EQ(index_->Dim(), dim);

    NGT::Timer timer;
    timer.start();
    auto result = index_->Query(query_dataset, conf);
    timer.stop();
    std::cout << std::endl << "Query time: " << timer.time << std::endl;
    SaveIdsToFile(result, nq, k, "onng", 20, 5, 40);
    // AssertAnns(result, nq, k);
}

#if 0

TEST_P(NGTONNGTest, ngtonng_delete) {
    assert(!xb.empty());

    index_->BuildAll(base_dataset, conf);  // Train + Add
    ASSERT_EQ(index_->Count(), nb);
    ASSERT_EQ(index_->Dim(), dim);

    faiss::ConcurrentBitsetPtr bitset = std::make_shared<faiss::ConcurrentBitset>(nb);
    for (auto i = 0; i < nq; ++i) {
        bitset->set(i);
    }

    auto result1 = index_->Query(query_dataset, conf);
    AssertAnns(result1, nq, k);

    index_->SetBlacklist(bitset);
    auto result2 = index_->Query(query_dataset, conf);
    AssertAnns(result2, nq, k, CheckMode::CHECK_NOT_EQUAL);
}

TEST_P(NGTONNGTest, ngtonng_serialize) {
    auto serialize = [](const std::string& filename, milvus::knowhere::BinaryPtr& bin, uint8_t* ret) {
        {
            // write and flush
            FileIOWriter writer(filename);
            writer(static_cast<void*>(bin->data.get()), bin->size);
        }

        FileIOReader reader(filename);
        reader(ret, bin->size);
    };

    {
        // serialize index
        index_->BuildAll(base_dataset, conf);
        auto binaryset = index_->Serialize(milvus::knowhere::Config());

        auto bin_obj_data = binaryset.GetByName("ngt_obj_data");
        std::string filename1 = "/tmp/ngt_obj_data_serialize.bin";
        auto load_data1 = new uint8_t[bin_obj_data->size];
        serialize(filename1, bin_obj_data, load_data1);

        auto bin_grp_data = binaryset.GetByName("ngt_grp_data");
        std::string filename2 = "/tmp/ngt_grp_data_serialize.bin";
        auto load_data2 = new uint8_t[bin_grp_data->size];
        serialize(filename2, bin_grp_data, load_data2);

        auto bin_prf_data = binaryset.GetByName("ngt_prf_data");
        std::string filename3 = "/tmp/ngt_prf_data_serialize.bin";
        auto load_data3 = new uint8_t[bin_prf_data->size];
        serialize(filename3, bin_prf_data, load_data3);

        auto bin_tre_data = binaryset.GetByName("ngt_tre_data");
        std::string filename4 = "/tmp/ngt_tre_data_serialize.bin";
        auto load_data4 = new uint8_t[bin_tre_data->size];
        serialize(filename4, bin_tre_data, load_data4);

        binaryset.clear();
        std::shared_ptr<uint8_t[]> obj_data(load_data1);
        binaryset.Append("ngt_obj_data", obj_data, bin_obj_data->size);

        std::shared_ptr<uint8_t[]> grp_data(load_data2);
        binaryset.Append("ngt_grp_data", grp_data, bin_grp_data->size);

        std::shared_ptr<uint8_t[]> prf_data(load_data3);
        binaryset.Append("ngt_prf_data", prf_data, bin_prf_data->size);

        std::shared_ptr<uint8_t[]> tre_data(load_data4);
        binaryset.Append("ngt_tre_data", tre_data, bin_tre_data->size);

        index_->Load(binaryset);
        ASSERT_EQ(index_->Count(), nb);
        ASSERT_EQ(index_->Dim(), dim);
        auto result = index_->Query(query_dataset, conf);
        AssertAnns(result, nq, conf[milvus::knowhere::meta::TOPK]);
    }
}
#endif
