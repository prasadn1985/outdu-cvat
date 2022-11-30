from cvat.settings.production import *

# add custom apps here
import ldap
from django_auth_ldap.config import LDAPSearch, GroupOfNamesType, NestedActiveDirectoryGroupType

IAM_TYPE = 'LDAP'
AUTH_LOGIN_NOTE = '''<p>
    For successful login please make sure you are member of cvat_users group
</p>'''

# Baseline configuration.
AUTH_LDAP_SERVER_URI = "ldap://test.outdu.org"

# Credentials for LDAP server
AUTH_LDAP_BIND_DN = "cn=admin,dc=test,dc=outdu,dc=org"
AUTH_LDAP_BIND_PASSWORD = "sr123"

# Set up basic user search
AUTH_LDAP_USER_SEARCH = LDAPSearch("ou=suborg1,dc=test,dc=outdu,dc=org",
    ldap.SCOPE_SUBTREE, "(uid=%(user)s)")

# Set up the basic group parameters.
AUTH_LDAP_GROUP_SEARCH = LDAPSearch("ou=suborg1,dc=test,dc=outdu,dc=org",
    ldap.SCOPE_SUBTREE, "(objectClass=*)")
AUTH_LDAP_GROUP_TYPE = GroupOfNamesType()

# Populate the Django user from the LDAP directory.
AUTH_LDAP_USER_ATTR_MAP = {
    "first_name": "cn",
    "last_name": "sn",
    "email": "mail",
}

#AUTH_LDAP_USER_FLAGS_BY_GROUP = {
#    "is_staff": "cn=cvat_admin,ou=suborg1,dc=test,dc=outdu,dc=org",
#    "is_superuser": "cn=cvat_admin,ou=suborg1,dc=test,dc=outdu,dc=org",
#}

AUTH_LDAP_ALWAYS_UPDATE_USER = True
AUTH_LDAP_FIND_GROUP_PERMS = True
# Cache group memberships for an hour to minimize LDAP traffic
AUTH_LDAP_CACHE_GROUPS = True
AUTH_LDAP_GROUP_CACHE_TIMEOUT = 3600
AUTH_LDAP_AUTHORIZE_ALL_USERS = True

#AUTH_LDAP_MIRROR_GROUPS = True

# Keep ModelBackend around for per-user permissions and maybe a local
# superuser.
AUTHENTICATION_BACKENDS += ['django_auth_ldap.backend.LDAPBackend']

# example 'cn=cvat_admin,cn=groups,cn=accounts,dc=example,dc=com'
# change your cn to match whatever groups you have in your LDAP
AUTH_LDAP_ADMIN_GROUPS = [
    'cn=cvat_admin,ou=suborg1,dc=test,dc=outdu,dc=org',
]
AUTH_LDAP_USER_GROUPS = [
    'cn=cvat_user,ou=suborg1,dc=test,dc=outdu,dc=org',
]
AUTH_LDAP_WORKER_GROUPS = [
    'cn=cvat_worker,ou=suborg1,dc=test,dc=outdu,dc=org',
]
AUTH_LDAP_BUSINESS_GROUPS = [
    'cn=cvat_business,ou=suborg1,dc=test,dc=outdu,dc=org',
]

DJANGO_AUTH_LDAP_GROUPS = {
        "admin": AUTH_LDAP_ADMIN_GROUPS, 
        "business": AUTH_LDAP_BUSINESS_GROUPS, 
        "user": AUTH_LDAP_USER_GROUPS, 
        "worker": AUTH_LDAP_WORKER_GROUPS, 
        }
