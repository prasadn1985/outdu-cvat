// Copyright (C) 2021-2022 Intel Corporation
//
// SPDX-License-Identifier: MIT

import './styles.scss';
import React, { useState, useEffect, useCallback } from 'react';
import Modal from 'antd/lib/modal';
import Notification from 'antd/lib/notification';
import { useSelector, useDispatch } from 'react-redux';
import { DownloadOutlined, LoadingOutlined } from '@ant-design/icons';
import Text from 'antd/lib/typography/Text';
import Select from 'antd/lib/select';
import Checkbox from 'antd/lib/checkbox';
import Input from 'antd/lib/input';
import Form from 'antd/lib/form';

import { CombinedState } from 'reducers/interfaces';
import { exportActions, exportDatasetAsync } from 'actions/export-actions';
import { getCloudStoragesAsync } from 'actions/cloud-storage-actions';
import getCore from 'cvat-core-wrapper';

const core = getCore();

type FormValues = {
    selectedFormat: string | undefined;
    saveImages: boolean;
    customName: string | undefined;
    cloudStorageId: number | undefined;
    cloudStorageDir: string | undefined;
};

function ExportDatasetModal(): JSX.Element {
    const [instanceType, setInstanceType] = useState('');
    const [activities, setActivities] = useState<string[]>([]);
    const [manifestPaths, setManifestPaths] = useState<string[]>([]);
    const [form] = Form.useForm();
    const dispatch = useDispatch();
    const instance = useSelector((state: CombinedState) => state.export.instance);
    const modalVisible = useSelector((state: CombinedState) => state.export.modalVisible);
    const dumpers = useSelector((state: CombinedState) => state.formats.annotationFormats.dumpers);
    const storages = useSelector((state: CombinedState) => state.cloudStorages.current);
    const { tasks: taskExportActivities, projects: projectExportActivities } = useSelector(
        (state: CombinedState) => state.export,
    );
    const query = useSelector((state: CombinedState) => state.cloudStorages.gettingQuery);
    useEffect(() => {
        dispatch(getCloudStoragesAsync({ ...query }));
    }, []);

    console.log("storages=",storages);
    const initActivities = (): void => {
        if (instance instanceof core.classes.Project) {
            setInstanceType(`project #${instance.id}`);
            setActivities(projectExportActivities[instance.id] || []);
        } else if (instance) {
            const taskID = instance instanceof core.classes.Task ? instance.id : instance.taskId;
            setInstanceType(`task #${taskID}`);
            setActivities(taskExportActivities[taskID] || []);
            if (instance.mode === 'interpolation' && instance.dimension === '2d') {
                form.setFieldsValue({ selectedFormat: 'CVAT for video 1.1' });
            } else if (instance.mode === 'annotation' && instance.dimension === '2d') {
                form.setFieldsValue({ selectedFormat: 'CVAT for images 1.1' });
            }
        }
    };

    useEffect(() => {
        initActivities();
    }, [instance?.id, instance instanceof core.classes.Project, taskExportActivities, projectExportActivities]);

    const closeModal = (): void => {
        form.resetFields();
        dispatch(exportActions.closeExportModal());
    };

    const handleExport = useCallback(
        (values: FormValues): void => {
            // have to validate format before so it would not be undefined
            console.log("####export-dataset-model cloudStorageId=", values.cloudStorageId)
            dispatch(
                exportDatasetAsync(
                    instance,
                    values.selectedFormat as string,
                    values.customName ? `${values.customName}.zip` : '',
                    values.saveImages,
                    values.cloudStorageId,
                    values.cloudStorageDir,
                ),
            );
            closeModal();
            Notification.info({
                message: 'Dataset export started',
                description:
                    `Dataset export was started for ${instanceType}. ` +
                    'Download will start automatically as soon as the dataset is ready.',
                className: `cvat-notification-notice-export-${instanceType.split(' ')[0]}-start`,
            });
        },
        [instance, instanceType],
    );

    const cloudStorageChangeHandler = cloudStorageId => {
        console.log("###cloudStorageChange=",cloudStorageId);
        const result_storage = storages.find(obj => {
            return obj.id === cloudStorageId;
        });
        console.log("result_storage=", result_storage);
        let manifest_paths=[];
        manifest_paths.push("/");
        for(let manifest_path of result_storage.manifests) {
            manifest_paths.push(manifest_path.substring(0, manifest_path.lastIndexOf("/")));
        }
        console.log("manifest_paths=", manifest_paths);
        setManifestPaths(manifest_paths);
    }

    return (
        <Modal
            title={`Export ${instanceType} as a dataset`}
            visible={modalVisible}
            onCancel={closeModal}
            onOk={() => form.submit()}
            className={`cvat-modal-export-${instanceType.split(' ')[0]}`}
            destroyOnClose
        >
            <Form
                name='Export dataset'
                form={form}
                labelCol={{ span: 8 }}
                wrapperCol={{ span: 16 }}
                initialValues={
                    {
                        selectedFormat: undefined,
                        saveImages: false,
                        customName: undefined,
                        cloudStorageId: undefined,
                        cloudStorageDir: undefined,
                    } as FormValues
                }
                onFinish={handleExport}
            >
                <Form.Item
                    name='selectedFormat'
                    label='Export format'
                    rules={[{ required: true, message: 'Format must be selected' }]}
                >
                    <Select virtual={false} placeholder='Select dataset format' className='cvat-modal-export-select'>
                        {dumpers
                            .sort((a: any, b: any) => a.name.localeCompare(b.name))
                            .filter((dumper: any): boolean => dumper.dimension === instance?.dimension)
                            .map(
                                (dumper: any): JSX.Element => {
                                    const pending = (activities || []).includes(dumper.name);
                                    const disabled = !dumper.enabled || pending;
                                    return (
                                        <Select.Option
                                            value={dumper.name}
                                            key={dumper.name}
                                            disabled={disabled}
                                            className='cvat-modal-export-option-item'
                                        >
                                            <DownloadOutlined />
                                            <Text disabled={disabled}>{dumper.name}</Text>
                                            {pending && <LoadingOutlined style={{ marginLeft: 10 }} />}
                                        </Select.Option>
                                    );
                                },
                            )}
                    </Select>
                </Form.Item>
                <Form.Item name='saveImages' valuePropName='checked' wrapperCol={{ offset: 8, span: 16 }}>
                    <Checkbox>Save images</Checkbox>
                </Form.Item>
                <Form.Item label='Custom name' name='customName'>
                    <Input
                        placeholder='Custom name for a dataset'
                        suffix='.zip'
                        className='cvat-modal-export-filename-input'
                    />
                </Form.Item>
                <Form.Item
                    name='cloudStorageId'
                    label='CloudStorage'
                    rules={[{ required: true, message: 'A bucket must be selected' }]} 
                >
                <Select virtual={false} placeholder='Select a bucket' 
                className='cvat-modal-export-select' onChange={cloudStorageChangeHandler}>
                        {storages
                            // .sort((a: any, b: any) => a.name.localeCompare(b.name))
                            // .filter((storage: any): boolean => storage.dimension === instance?.dimension)
                            .map(
                                (storage: any): JSX.Element => {
                                    const pending = (activities || []).includes(storage.displayName);
                                    // const disabled = !storage.enabled || pending;
                                    const disabled = false;
                                    return (
                                        <Select.Option
                                            value={storage.id}
                                            key={storage.displayName}
                                            disabled={disabled}
                                            className='cvat-modal-export-option-item'
                                        >
                                            <DownloadOutlined />
                                            <Text disabled={disabled}>{storage.displayName}</Text>
                                            {pending && <LoadingOutlined style={{ marginLeft: 10 }} />}
                                        </Select.Option>
                                    );
                                },
                            )}
                    </Select>                
                </Form.Item>
                <Form.Item
                    name='cloudStorageDir'
                    label='CloudStorageDir'
                    rules={[{ required: true, message: 'A directory must be selected' }]} 
                >
                <Select virtual={false} placeholder='Select a directory' 
                className='cvat-modal-export-select'>
                        {manifestPaths
                            .map(
                                (manifest_path: any): JSX.Element => {
                                    const disabled = false;
                                    return (
                                        <Select.Option
                                            value={manifest_path}
                                            key={manifest_path}
                                            disabled={disabled}
                                            className='cvat-modal-export-option-item'
                                        >
                                            <DownloadOutlined />
                                            <Text disabled={disabled}>{manifest_path}</Text>
                                        </Select.Option>
                                    );
                                },
                            )}
                    </Select>
                
                </Form.Item>
            </Form>
        </Modal>
    );
}

export default React.memo(ExportDatasetModal);
