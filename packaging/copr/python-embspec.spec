%global pypi_name embspec
%global pypi_module embspec

Name:           python-%{pypi_name}
Version:        0.1.0
Release:        1%{?dist}
Summary:        Embedding pipeline ops + drift detection for production RAG

License:        Apache-2.0
URL:            https://github.com/MukundaKatta/%{pypi_name}
Source0:        https://github.com/MukundaKatta/%{pypi_name}/releases/download/v%{version}/%{pypi_module}-%{version}.tar.gz

BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  python3dist(uv-build)

%global _description %{expand:
Three primitives that close the embedding-pipeline-ops gap left between
Evidently (tabular-ML drift heritage), Phoenix (full observability platform),
and the Drift-Adapter paper (research code): IndexManifest version
assertions, DriftAdapter for in-place model migrations, neighbor-stability
eval on a frozen probe set.}

%description %_description

%package -n python3-%{pypi_name}
Summary:        %{summary}
Requires:       python3-numpy >= 1.24

%description -n python3-%{pypi_name} %_description

%prep
%autosetup -p1 -n %{pypi_module}-%{version}

%generate_buildrequires
%pyproject_buildrequires

%build
%pyproject_wheel

%install
%pyproject_install
%pyproject_save_files %{pypi_module}

%check
%pyproject_check_import

%files -n python3-%{pypi_name} -f %{pyproject_files}
%license LICENSE
%doc README.md CHANGELOG.md

%changelog
* Sat May 09 2026 Mukunda Katta <mukunda.vjcs6@gmail.com> - 0.1.0-1
- Initial Fedora packaging
