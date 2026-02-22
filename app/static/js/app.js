/* AngularJS MVC Application - DSPy Classification Studio */
(function() {
'use strict';

var app = angular.module('classifierApp', []);

/* ===== HistoryService (localStorage persistence) ===== */
app.factory('HistoryService', function() {
    var KEY = 'classifier_history';
    return {
        getAll: function() {
            try { return JSON.parse(localStorage.getItem(KEY)) || []; }
            catch(e) { return []; }
        },
        add: function(item) {
            var history = this.getAll();
            history.unshift(item);
            if (history.length > 50) history = history.slice(0, 50);
            localStorage.setItem(KEY, JSON.stringify(history));
        },
        clear: function() { localStorage.removeItem(KEY); }
    };
});

/* ===== ClassifierService (API communication) ===== */
app.factory('ClassifierService', ['$http', function($http) {
    var BASE = '/api';
    return {
        classify: function(text, type) {
            return $http.post(BASE + '/classify', {text: text, classifier_type: type});
        },
        batchClassify: function(texts, type) {
            return $http.post(BASE + '/classify/batch', {texts: texts, classifier_type: type || 'sentiment'});
        },
        runAgent: function(text) {
            return $http.post(BASE + '/agent/analyze', {text: text, enable_knowledge_graph: true});
        },
        getKnowledgeGraph: function() {
            return $http.get(BASE + '/knowledge-graph');
        },
        seedKnowledgeGraph: function() {
            return $http.post(BASE + '/knowledge-graph/seed');
        },
        inferGraph: function(query) {
            return $http.post(BASE + '/graph/infer', query);
        },
        getClassifiers: function() {
            return $http.get(BASE + '/classifiers');
        },
        getHealth: function() {
            return $http.get('/health');
        }
    };
}]);

/* ===== MainController ===== */
app.controller('MainController', ['$scope', 'ClassifierService', 'HistoryService',
function($scope, ClassifierService, HistoryService) {
    $scope.currentTab = 'classify';
    $scope.classifierType = 'sentiment';
    $scope.inputText = '';
    $scope.batchText = '';
    $scope.agentText = '';
    $scope.loading = false;
    $scope.error = null;
    $scope.classifyResult = null;
    $scope.batchResult = null;
    $scope.agentResult = null;
    $scope.graphData = null;
    $scope.graphInference = null;
    $scope.graphSubTab = 'explore';
    $scope.entitySearch = '';
    $scope.graphQuery = {
        entity: '',
        entity_type: '',
        max_depth: 2,
        relation_filter: ''
    };
    $scope.history = HistoryService.getAll();

    /* ===== Graph helper functions ===== */
    $scope.getEntityTypes = function() {
        if (!$scope.graphData || !$scope.graphData.nodes) return [];
        var types = {};
        $scope.graphData.nodes.forEach(function(n) {
            types[n.type] = true;
        });
        return Object.keys(types).sort();
    };

    $scope.getEntitiesByType = function(entityType) {
        if (!$scope.graphData || !$scope.graphData.nodes) return [];
        return $scope.graphData.nodes.filter(function(n) {
            return n.type === entityType;
        }).sort(function(a, b) {
            return a.name.localeCompare(b.name);
        });
    };

    $scope.getRelationTypes = function() {
        if (!$scope.graphData || !$scope.graphData.edges) return [];
        var types = {};
        $scope.graphData.edges.forEach(function(e) {
            types[e.relation] = true;
        });
        return Object.keys(types).sort();
    };

    $scope.quickInfer = function(name, entityType) {
        $scope.graphQuery.entity = name;
        $scope.graphQuery.entity_type = entityType || '';
        $scope.graphSubTab = 'infer';
        $scope.runGraphInference();
    };

    $scope.classify = function() {
        if (!$scope.inputText.trim()) return;
        $scope.loading = true;
        $scope.error = null;
        ClassifierService.classify($scope.inputText, $scope.classifierType)
            .then(function(resp) {
                $scope.classifyResult = resp.data;
                HistoryService.add(resp.data);
                $scope.history = HistoryService.getAll();
            })
            .catch(function(err) {
                $scope.error = (err.data && err.data.error) ? err.data.error : 'Request failed';
            })
            .finally(function() { $scope.loading = false; });
    };

    $scope.batchClassify = function() {
        if (!$scope.batchText.trim()) return;
        var texts = $scope.batchText.split('\n').filter(function(t) { return t.trim(); });
        if (!texts.length) return;
        $scope.loading = true;
        $scope.error = null;
        ClassifierService.batchClassify(texts)
            .then(function(resp) { $scope.batchResult = resp.data; })
            .catch(function(err) {
                $scope.error = (err.data && err.data.error) ? err.data.error : 'Request failed';
            })
            .finally(function() { $scope.loading = false; });
    };

    $scope.runAgent = function() {
        if (!$scope.agentText.trim()) return;
        $scope.loading = true;
        $scope.error = null;
        ClassifierService.runAgent($scope.agentText)
            .then(function(resp) {
                $scope.agentResult = resp.data;
                HistoryService.add({classifier_type: 'agent', text: $scope.agentText, result: resp.data});
                $scope.history = HistoryService.getAll();
            })
            .catch(function(err) {
                $scope.error = (err.data && err.data.error) ? err.data.error : 'Request failed';
            })
            .finally(function() { $scope.loading = false; });
    };

    $scope.loadKnowledgeGraph = function() {
        ClassifierService.getKnowledgeGraph()
            .then(function(resp) { $scope.graphData = resp.data; })
            .catch(function() { $scope.error = 'Failed to load knowledge graph'; });
    };

    $scope.seedKnowledgeGraph = function() {
        $scope.loading = true;
        $scope.error = null;
        ClassifierService.seedKnowledgeGraph()
            .then(function() {
                $scope.loadKnowledgeGraph();
            })
            .catch(function(err) {
                $scope.error = (err.data && err.data.error) ? err.data.error : 'Failed to seed graph';
            })
            .finally(function() { $scope.loading = false; });
    };

    $scope.runGraphInference = function() {
        if (!$scope.graphQuery.entity || !$scope.graphQuery.entity.trim()) {
            $scope.error = 'Entity is required for graph inference.';
            return;
        }

        $scope.loading = true;
        $scope.error = null;
        var query = {
            entity: $scope.graphQuery.entity,
            max_depth: Number($scope.graphQuery.max_depth || 2)
        };
        if ($scope.graphQuery.entity_type && $scope.graphQuery.entity_type.trim()) {
            query.entity_type = $scope.graphQuery.entity_type;
        }
        if ($scope.graphQuery.relation_filter && $scope.graphQuery.relation_filter.trim()) {
            query.relation_filter = $scope.graphQuery.relation_filter;
        }

        ClassifierService.inferGraph(query)
            .then(function(resp) { $scope.graphInference = resp.data; })
            .catch(function(err) {
                $scope.error = (err.data && err.data.error) ? err.data.error : 'Graph inference failed';
            })
            .finally(function() { $scope.loading = false; });
    };

    $scope.clearHistory = function() {
        HistoryService.clear();
        $scope.history = [];
    };
}]);

})();
