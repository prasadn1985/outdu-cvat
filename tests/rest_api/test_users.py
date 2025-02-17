# Copyright (C) 2021-2022 Intel Corporation
# Copyright (C) 2022 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

from http import HTTPStatus
import json
import typing

import pytest
from deepdiff import DeepDiff

from rest_api.utils.config import make_api_client


@pytest.mark.usefixtures('dontchangedb')
class TestGetUsers:
    def _test_can_see(self, user, data,
            id_: typing.Union[typing.Literal['self'], int, None] = None,
            *,
            exclude_paths='', **kwargs):
        with make_api_client(user) as api_client:
            # TODO: refactor into several functions
            if id_ == 'self':
                (_, response) = api_client.users_api.retrieve_self(**kwargs,
                    _parse_response=False)
                assert response.status == HTTPStatus.OK
                response_data = json.loads(response.data)
            elif id_ is None:
                fetch_all = kwargs.get('page_size') == 'all'
                if fetch_all:
                    kwargs.pop('page_size')

                (_, response) = api_client.users_api.list(**kwargs, _parse_response=False)
                assert response.status == HTTPStatus.OK
                parsed_data = json.loads(response.data)
                response_data = parsed_data.get('results', [])

                if fetch_all:
                    page_idx = 2
                    while parsed_data.get('next'):
                        (_, response) = api_client.users_api.list(**kwargs,
                            _parse_response=False, page=page_idx)
                        assert response.status == HTTPStatus.OK
                        parsed_data = json.loads(response.data)
                        response_data += parsed_data.get('results', [])
                        page_idx += 1
            else:
                (_, response) = api_client.users_api.retrieve(id_, **kwargs,
                    _parse_response=False)
                assert response.status == HTTPStatus.OK
                response_data = json.loads(response.data)

        assert DeepDiff(data, response_data, ignore_order=True,
            exclude_paths=exclude_paths) == {}

    def _test_cannot_see(self, user, id_: typing.Union[typing.Literal['self'], int, None] = None,
            **kwargs):
        with make_api_client(user) as api_client:
            # TODO: refactor into several functions
            if id_ == 'self':
                (_, response) = api_client.users_api.retrieve_self(**kwargs,
                    _parse_response=False, _check_status=False)
            elif id_ is None:
                (_, response) = api_client.users_api.list(**kwargs,
                    _parse_response=False, _check_status=False)
            else:
                (_, response) = api_client.users_api.retrieve(id_, **kwargs,
                    _parse_response=False, _check_status=False)
            assert response.status == HTTPStatus.FORBIDDEN

    def test_admin_can_see_all_others(self, users):
        exclude_paths = [f"root[{i}]['last_login']" for i in range(len(users))]
        self._test_can_see('admin2', users.raw, exclude_paths=exclude_paths,
            page_size="all")

    def test_everybody_can_see_self(self, users_by_name):
        for user, data in users_by_name.items():
            self._test_can_see(user, data, id_="self", exclude_paths="root['last_login']")

    def test_non_members_cannot_see_list_of_members(self):
        self._test_cannot_see('user2', org='org1')

    def test_non_admin_cannot_see_others(self, users):
        non_admins = (v for v in users if not v['is_superuser'])
        user = next(non_admins)['username']
        user_id = next(non_admins)['id']

        self._test_cannot_see(user, id_=user_id)

    def test_all_members_can_see_list_of_members(self, find_users, users):
        org_members = [user['username'] for user in find_users(org=1)]
        available_fields = ['url', 'id', 'username', 'first_name', 'last_name']

        data = [dict(filter(lambda row: row[0] in available_fields, user.items()))
            for user in users if user['username'] in org_members]

        for member in org_members:
            self._test_can_see(member, data, org='org1')
