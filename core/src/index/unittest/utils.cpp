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

#include "unittest/utils.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"

#include <bits/stdint-intn.h>
#include <gtest/gtest.h>
#include <math.h>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

INITIALIZE_EASYLOGGINGPP

const char* base_file = "/root/sift/sift_base.tsv";
const char* query_file = "/root/sift/sift_query.tsv";

void
InitLog() {
    el::Configurations defaultConf;
    defaultConf.setToDefault();
    defaultConf.set(el::Level::Debug, el::ConfigurationType::Format, "[%thread-%datetime-%level]: %msg (%fbase:%line)");
    el::Loggers::reconfigureLogger("default", defaultConf);
}

void
DataGen::Init_with_default(const bool is_binary) {
    Generate(dim, nb, nq, is_binary);
}

void
DataGen::Generate(const int dim, const int nb, const int nq, const bool is_binary) {
    this->dim = dim;
    this->nb = nb;
    this->nq = nq;

    if (!is_binary) {
        GenAll(dim, nb, xb, ids, xids, nq, xq);
        assert(xb.size() == (size_t)dim * nb);
        assert(xq.size() == (size_t)dim * nq);

        base_dataset = milvus::knowhere::GenDatasetWithIds(nb, dim, xb.data(), ids.data());
        query_dataset = milvus::knowhere::GenDataset(nq, dim, xq.data());
    } else {
        int64_t dim_x = dim / 8;
        GenAll(dim_x, nb, xb_bin, ids, xids, nq, xq_bin);
        assert(xb_bin.size() == (size_t)dim_x * nb);
        assert(xq_bin.size() == (size_t)dim_x * nq);

        base_dataset = milvus::knowhere::GenDatasetWithIds(nb, dim, xb_bin.data(), ids.data());
        query_dataset = milvus::knowhere::GenDataset(nq, dim, xq_bin.data());
    }

    id_dataset = milvus::knowhere::GenDatasetWithIds(nq, dim, nullptr, ids.data());
    xid_dataset = milvus::knowhere::GenDatasetWithIds(nq, dim, nullptr, xids.data());
}

void
DataGen::GenerateSift1M(const int dim, const int nb, const int nq) {
    this->dim = dim;
    this->nb = nb;
    this->nq = nq;

    GenAllSift1M(dim, nb, xb, ids, xids, nq, xq);
    assert(xb.size() == (size_t)dim * nb);
    assert(xq.size() == (size_t)dim * nq);

    base_dataset = milvus::knowhere::GenDatasetWithIds(nb, dim, xb.data(), ids.data());
    query_dataset = milvus::knowhere::GenDataset(nq, dim, xq.data());

    id_dataset = milvus::knowhere::GenDatasetWithIds(nq, dim, nullptr, ids.data());
    xid_dataset = milvus::knowhere::GenDatasetWithIds(nq, dim, nullptr, xids.data());
}

void
GenAll(const int64_t dim, const int64_t nb, std::vector<float>& xb, std::vector<int64_t>& ids,
       std::vector<int64_t>& xids, const int64_t nq, std::vector<float>& xq) {
    xb.resize(nb * dim);
    xq.resize(nq * dim);
    ids.resize(nb);
    xids.resize(1);
    GenBase(dim, nb, xb.data(), ids.data(), nq, xq.data(), xids.data(), false);
}

void
GenAllSift1M(const int64_t dim, const int64_t nb, std::vector<float>& xb, std::vector<int64_t>& ids,
             std::vector<int64_t>& xids, const int64_t nq, std::vector<float>& xq) {
    xb.resize(nb * dim);
    xq.resize(nq * dim);
    ids.resize(nb);
    xids.resize(1);
    GenData(dim, xb.data(), ids.data(), xq.data(), xids.data());
}

void
GenData(const int64_t dim, float* xb, int64_t* ids, float* xq, int64_t* xids) {
    xids[0] = 3;
    ReadData(base_file, 128, xb);
    ReadData(query_file, 128, xq);
    for (size_t i = 0; i < 1000000; ++i) {
        ids[i] = i;
    }
}

void
ReadData(const char* file_name, const int64_t dimension, float* data) {
    std::ifstream file;
    file.open(file_name);
    if (!file.is_open())
    {
        std::cout << "Open file " << file_name << " failed" << std::endl;
        exit(-1);
    }
    std::string s;
    int index = 0;
    const char * start;
    char * end;

    size_t size;
    while (std::getline(file, s))
    {
        start = s.c_str();
        size = 0;
        for (float f = std::strtof(start, &end); start != end && size++ < dimension; f = std::strtof(start, &end))
        {
            data[index++] = f;
            start = end;
        }
    }
    file.close();
}

void
GenAll(const int64_t dim, const int64_t nb, std::vector<uint8_t>& xb, std::vector<int64_t>& ids,
       std::vector<int64_t>& xids, const int64_t nq, std::vector<uint8_t>& xq) {
    xb.resize(nb * dim);
    xq.resize(nq * dim);
    ids.resize(nb);
    xids.resize(1);
    GenBase(dim, nb, xb.data(), ids.data(), nq, xq.data(), xids.data(), true);
}

void
GenBase(const int64_t dim, const int64_t nb, const void* xb, int64_t* ids, const int64_t nq, const void* xq,
        int64_t* xids, bool is_binary) {
    if (!is_binary) {
        float* xb_f = (float*)xb;
        float* xq_f = (float*)xq;
        for (auto i = 0; i < nb; ++i) {
            for (auto j = 0; j < dim; ++j) {
                xb_f[i * dim + j] = drand48();
            }
            xb_f[dim * i] += i / 1000.;
            ids[i] = i;
        }
        for (int64_t i = 0; i < nq * dim; ++i) {
            xq_f[i] = xb_f[i];
        }
    } else {
        uint8_t* xb_u = (uint8_t*)xb;
        uint8_t* xq_u = (uint8_t*)xq;
        for (auto i = 0; i < nb; ++i) {
            for (auto j = 0; j < dim; ++j) {
                xb_u[i * dim + j] = (uint8_t)lrand48();
            }
            xb_u[dim * i] += i / 1000.;
            ids[i] = i;
        }
        for (int64_t i = 0; i < nq * dim; ++i) {
            xq_u[i] = xb_u[i];
        }
    }
    xids[0] = 3;  // pseudo random
}

FileIOReader::FileIOReader(const std::string& fname) {
    name = fname;
    fs = std::fstream(name, std::ios::in | std::ios::binary);
}

FileIOReader::~FileIOReader() {
    fs.close();
}

size_t
FileIOReader::operator()(void* ptr, size_t size) {
    fs.read(reinterpret_cast<char*>(ptr), size);
    return size;
}

FileIOWriter::FileIOWriter(const std::string& fname) {
    name = fname;
    fs = std::fstream(name, std::ios::out | std::ios::binary);
}

FileIOWriter::~FileIOWriter() {
    fs.close();
}

size_t
FileIOWriter::operator()(void* ptr, size_t size) {
    fs.write(reinterpret_cast<char*>(ptr), size);
    return size;
}

void
AssertAnns(const milvus::knowhere::DatasetPtr& result, const int nq, const int k, const CheckMode check_mode) {
    auto ids = result->Get<int64_t*>(milvus::knowhere::meta::IDS);
    for (auto i = 0; i < nq; i++) {
        switch (check_mode) {
            case CheckMode::CHECK_EQUAL:
                ASSERT_EQ(i, *((int64_t*)(ids) + i * k));
                break;
            case CheckMode::CHECK_NOT_EQUAL:
                ASSERT_NE(i, *((int64_t*)(ids) + i * k));
                break;
            default:
                ASSERT_TRUE(false);
                break;
        }
    }
}

void
SaveIdsToFile(const milvus::knowhere::DatasetPtr& result, const int nq, const int k, const char* algo, int edge_size,
              int arg1, int arg2) {
    auto ids = result->Get<int64_t*>(milvus::knowhere::meta::IDS);
    std::string output("/root/output/");
    output = output + algo + "-" + std::to_string(edge_size) + "-" + std::to_string(arg1) + "-" + std::to_string(arg2);
    std::ofstream file(output);
    if (file.is_open()) {
        for (size_t i = 0; i < nq; ++i) {
            for (size_t j = 0; j < k; ++j) {
                file << *((int64_t*)(ids) + i * k + j);
                if (j < k - 1) {
                    file << "\t";
                    continue;
                }
                file << "\n";
            }
        }
        file.close();
    } else {
        std::cout << "Open file failed" << std::endl;
    }
}

void
PrintResult(const milvus::knowhere::DatasetPtr& result, const int& nq, const int& k) {
    auto ids = result->Get<int64_t*>(milvus::knowhere::meta::IDS);
    auto dist = result->Get<float*>(milvus::knowhere::meta::DISTANCE);

    std::stringstream ss_id;
    std::stringstream ss_dist;
    for (auto i = 0; i < nq; i++) {
        for (auto j = 0; j < k; ++j) {
            // ss_id << *(ids->data()->GetValues<int64_t>(1, i * k + j)) << " ";
            // ss_dist << *(dists->data()->GetValues<float>(1, i * k + j)) << " ";
            ss_id << *((int64_t*)(ids) + i * k + j) << " ";
            ss_dist << *((float*)(dist) + i * k + j) << " ";
        }
        ss_id << std::endl;
        ss_dist << std::endl;
    }
    std::cout << "id\n" << ss_id.str() << std::endl;
    std::cout << "dist\n" << ss_dist.str() << std::endl;
}
